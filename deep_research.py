import concurrent.futures, logging, re, itertools
from typing import List, Set
from sentence_transformers import SentenceTransformer
from duckduckgo_search import DDGS
from vector_store import VectorStore
from planning import make_plan
from web_scraper import crawl_site, clean

log = logging.getLogger(__name__)
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def chunk(text: str, max_len=500) -> List[str]:
    out, cur = [], ""
    for s in SENT_SPLIT.split(text):
        if len(cur) + len(s) > max_len:
            out.append(cur.strip())
            cur = s + " "
        else:
            cur += s + " "
    if cur.strip():
        out.append(cur.strip())
    return out


class DeepResearcher:
    def __init__(self, llm, question: str,
                 site: str | None,
                 max_tasks: int = 3,
                 max_iters: int = 2):
        self.llm = llm
        self.question, self.site = question, site
        self.max_tasks, self.max_iters = max_tasks, max_iters
        self.embedder = SentenceTransformer(EMBED_MODEL)
        self.db = VectorStore(self.embedder.get_sentence_embedding_dimension())
        self._visited: Set[str] = set()     # pour les logs

    # ---------------- recherche / collecte -----------------
    @staticmethod
    def _search_ddg(query: str, k=8) -> List[str]:
        with DDGS() as ddg:
            return [r["href"] for r in ddg.text(query, max_results=k)]

    def _collect_docs(self, subq: str) -> List[str]:
        if self.site:
            return crawl_site(self.site, limit_pages=40, max_depth=2)
        log.info("🔍 DuckDuckGo query → %s", subq)
        urls = self._search_ddg(subq)
        import requests
        docs = []
        for u in urls:
            if u in self._visited:
                continue
            self._visited.add(u)
            try:
                html = requests.get(
                    u, timeout=10,
                    headers={"User-Agent": "Mozilla/5.0"}
                ).text
                docs.append(clean(html))
            except Exception:
                pass
        return docs

    # ----- résumé du contexte déjà indexé -----
    def _coverage(self, k: int = 12) -> str:
        qv = self.embedder.encode([self.question])[0].reshape(1, -1)
        ctx = "\n\n".join(t for t, _ in self.db.search(qv, k=k))
        prompt = (
            "En 5 phrases maxi, résume le CONTEXTE (extraits) ci‑dessous:\n"
            f"{ctx}\n\nRÉSUMÉ:"
        )
        return self.llm.chat([{"role": "user", "content": prompt}])

    # ----- deux points encore flous -----
    def _gaps(self, coverage: str) -> list[str]:
        prompt = (
            "Voici un résumé partiel d'une recherche.\n"
            f"{coverage}\n\n"
            "Donne 2 aspects IMPORTANTS qui ne sont pas encore couverts, "
            "sous forme de puces courtes."
        )
        out = self.llm.chat([{"role": "user", "content": prompt}])
        return [l.strip("•- ") for l in out.splitlines() if l.strip()]

    # ---------------- boucle itérative -----------------
    def run(self) -> str:
        done: list[str] = []      # sous‑questions déjà traitées
        context = ""              # résumé courant

        for it in range(1, self.max_iters + 1):
            log.info("\n=== ITERATION %d : PLANNING ===", it)

            # ---------- planification ----------
            subtasks = make_plan(
                self.llm, self.question, self.max_tasks,
                context=context, done=done, gaps=[]        # gaps vide au 1er tour
            )
            if not subtasks:
                log.info("Pas de nouvelles sous‑questions ➜ stop.")
                break

            for idx, s in enumerate(subtasks, 1):
                log.info("PLAN %d.%d ➜ %s", it, idx, s)

            # ---------- collecte ----------
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(subtasks)) as ex:
                ex.map(self._process_task, subtasks)

            # ---------- logs URLs ----------
            log.info("--- VISITED URLS (%d) ---", len(self._visited))
            for u in itertools.islice(self._visited, 10):
                log.info("· %s", u)
            if len(self._visited) > 10:
                log.info("… %d autres", len(self._visited) - 10)

            # ---------- coverage + gaps ----------
        # ---------- coverage + gaps ----------
            coverage = self._coverage()
            log.info("--- COVERAGE RESUME ---\n%s", coverage)

            gaps = self._gaps(coverage)[:2]
            import re
            _CLEAN = re.compile(r"^[•*\\-\\s]+")
            gaps = [_CLEAN.sub("", g).strip() for g in gaps]   # nettoie bullet points
            gaps = [g for g in gaps if "aspects importants" not in g.lower()]  # filtre les phrases bateaux

            if gaps:
                log.info("--- UNKNOWN ASPECTS ---")
                for g in gaps:
                    log.info("❓ %s", g)

                # <--  NOUVEAU  -->  injecte les gaps comme futures sous-questions
                # Stocke-les pour la boucle SUIVANTE
                done.extend(subtasks)          # déjà traitées
                # convertit les gaps en sous-questions pour la prochaine planif
                subtasks = [g for g in gaps if g not in done]
                if subtasks:
                    log.info("Injecte gaps comme nouvelles sous-questions : %s", subtasks)
                    with concurrent.futures.ThreadPoolExecutor(max_workers=len(subtasks)) as ex:
                        ex.map(self._process_task, subtasks)
                    done.extend(subtasks)      # marque aussi comme traitées

                context = self._coverage()     # résumé après l'injection
                continue 

            # ---------- prépare l’itération suivante ----------
            done.extend(subtasks)
            context = coverage

        # ---------- synthèse finale ----------
        log.info("=== FINISH TOUCH : SYNTHÈSE ===")
        answer = self._synth()
        log.info("=== DONE ===")
        return answer

    def _process_task(self, subq: str):
        docs = self._collect_docs(subq)
        ch = [c for d in docs for c in chunk(d)]
        if not ch:
            log.warning("Aucun contenu pour « %s »", subq)
            return
        vecs = self.embedder.encode(ch, batch_size=16, show_progress_bar=False)
        self.db.add(vecs, ch)
        log.info("Indexed %d chunks for « %s »", len(ch), subq)

    # ---------------- synthèse finale -----------------
    def _synth(self) -> str:
        qvec = self.embedder.encode([self.question])[0].reshape(1, -1)
        ctx = "\n\n".join(t for t, _ in self.db.search(qvec, k=20))
        prompt = (
            "Réponds à la QUESTION en markdown (citations 1,2…).\n"
            f"QUESTION: {self.question}\nCONTEXTE:\n{ctx}\n\nRéponse détaillée:"
        )
        return self.llm.chat([{"role": "user", "content": prompt}])