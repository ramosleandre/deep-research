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

    # ---------------- boucle itérative -----------------
    def run(self) -> str:
        for it in range(1, self.max_iters + 1):
            log.info("\n=== ITERATION %d : PLANNING ===", it)
            subtasks = make_plan(self.llm, self.question, self.max_tasks)
            for idx, s in enumerate(subtasks, 1):
                log.info("PLAN %d.%d ➜ %s", it, idx, s)

            with concurrent.futures.ThreadPoolExecutor(
                max_workers=len(subtasks)
            ) as ex:
                ex.map(self._process_task, subtasks)

            log.info("--- VISITED URLS (%d) ---", len(self._visited))
            for u in itertools.islice(self._visited, 10):
                log.info("· %s", u)
            if len(self._visited) > 10:
                log.info("… %d autres", len(self._visited) - 10)

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