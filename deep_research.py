import concurrent.futures, logging, re, itertools
from typing import List, Set
import unicodedata
from sentence_transformers import SentenceTransformer
from duckduckgo_search import DDGS

from vector_store import VectorStore
from planning import make_plan
from web_scraper import crawl_site, clean

log = logging.getLogger(__name__)

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def chunk(text: str, max_len: int = 500) -> List[str]:
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

def _normalize(q: str) -> str:
    # Apostrophe → espace, accents supprimés pour meilleure recherche
    q = q.replace("’", " ").replace("'", " ")
    q = unicodedata.normalize("NFKD", q).encode("ascii", "ignore").decode()
    return " ".join(q.split())

class DeepResearcher:
    def __init__(self, llm, question: str, site: str | None,
                 max_tasks: int = 3, max_iters: int = 2):
        self.llm = llm
        self.question, self.site = question, site

        # scrap limité à 2 tours si on cible un site
        self.scrap_rounds = 0
        self.scrap_limit = 2 if site else None

        self.max_tasks = max_tasks
        self.max_iters = 1 if site else max_iters   # 1 seule itération si scrape site

        self.embedder = SentenceTransformer(EMBED_MODEL)
        self.db = VectorStore(self.embedder.get_sentence_embedding_dimension())
        self._visited: Set[str] = set()

    # ------------------- collecte -------------------
    @staticmethod
    def _search_ddg(q: str, k: int = 8) -> List[str]:
        with DDGS() as ddg:
            return [r["href"] for r in ddg.text(q, max_results=k)]

    def _collect_docs(self, subq: str) -> List[str]:
        if self.site:                               # mode crawler
            if self.scrap_rounds >= self.scrap_limit:
                log.info("[SCRAPER] Limite atteinte → skip « %s »", subq)
                return []
            self.scrap_rounds += 1
            return crawl_site(self.site, limit_pages=40, max_depth=2)

        # recherche web
        log.info("[SEARCH] DuckDuckGo → %s", subq)
        urls = self._search_ddg(subq)
        import requests
        docs = []
        for u in urls:
            if u in self._visited:
                continue
            self._visited.add(u)
            try:
                html = requests.get(u, timeout=10,
                                    headers={"User-Agent": "Mozilla/5.0"}).text
                docs.append(clean(html))
            except Exception:
                pass
        return docs

    # ------------------- helpers LLM ----------------
    def _coverage(self, k: int = 12) -> str:
        qv = self.embedder.encode([self.question])[0].reshape(1, -1)
        ctx = "\n\n".join(t for t, _ in self.db.search(qv, k=k))
        prompt = f"En 5 phrases max, résume ce CONTEXTE:\n{ctx}\n\nRésumé:"
        return self.llm.chat([{"role": "user", "content": prompt}])

    def _gaps(self, coverage: str) -> list[str]:
        prompt = (coverage + "\n\nDonne 2 points IMPORTANTS non couverts "
                  "sous forme de puces.")
        out = self.llm.chat([{"role": "user", "content": prompt}])
        return [l.strip("•*- ").strip() for l in out.splitlines() if l.strip()]

    # ------------------- boucle principale ----------
    def run(self) -> str:
        done, context, gaps_prev = [], "", []

        for it in range(1, self.max_iters + 1):
            log.info("\n=== ITERATION %d : PLANNING ===", it)

            subtasks = make_plan(self.llm, self.question, self.max_tasks,
                                 context, done, gaps_prev)

            # fallback si planner vide
            if not subtasks:
                subtasks = gaps_prev or [self.question]
                gaps_prev = []
                log.info("[FALLBACK] Planner vide → %s", "gaps" if gaps_prev else "question brute")

            log.info("[PLANNING] Sous-questions:")
            for s in subtasks:
                log.info(" - %s", s)

            with concurrent.futures.ThreadPoolExecutor(max_workers=len(subtasks)) as ex:
                ex.map(self._process_task, subtasks)

            log.info("[VISITED WEBSITES] (%d)", len(self._visited))
            for u in itertools.islice(self._visited, 10):
                log.info(" - %s", u)

            coverage = self._coverage()
            log.info("[SUMMARY] %s", coverage.replace("\n", " "))

            gaps = self._gaps(coverage)[:2]
            _clean = re.compile(r"^[•*\\-\\s]+")
            gaps = [_clean.sub("", g).strip() for g in gaps]
            gaps = [g for g in gaps if g and g not in done]

            if gaps:
                log.info("[GAPS] À explorer:")
                for g in gaps:
                    log.info(" - %s", g)

            done.extend(subtasks)
            context = coverage
            gaps_prev = gaps

        log.info("\n=== FINISH TOUCH : SYNTHÈSE ===")
        return self._synth()

    # ------------------- worker ---------------------
    def _process_task(self, subq: str):
        docs = self._collect_docs(subq)
        chunks = [c for d in docs for c in chunk(d)]
        if not chunks:
            log.warning("[EMPTY] Aucun contenu pour « %s »", subq)
            return
        vecs = self.embedder.encode(chunks, batch_size=16, show_progress_bar=False)
        self.db.add(vecs, chunks)
        log.info("[INDEXED] %d chunks pour « %s »", len(chunks), subq)

    # ------------------- synthèse finale -----------
    def _synth(self) -> str:
        qv = self.embedder.encode([self.question])[0].reshape(1, -1)
        ctx_parts = self.db.search(qv, k=20)
        if ctx_parts:
            ctx = "\n\n".join(t for t, _ in ctx_parts)
            prompt = ("Réponds en markdown. QUESTION: {q}\nCONTEXTE:\n{c}\n\nRéponse:"
                      .format(q=self.question, c=ctx))
        else:   # aucun document indexé
            prompt = f"Réponds clairement en markdown à la question suivante : {self.question}"
        return self.llm.chat([{"role": "user", "content": prompt}])