import requests, re, urllib.parse, logging, collections
from bs4 import BeautifulSoup
from typing import List

log = logging.getLogger(__name__)
TAG_RE = re.compile(r"<[^>]+>")

def clean(html: str) -> str:
    txt = BeautifulSoup(html, "html.parser").get_text(" ", strip=True)
    return re.sub(r"\s+", " ", txt)

def crawl_site(root: str, limit_pages: int = 30, max_depth: int = 2) -> List[str]:
    """BFS interne au domaine root (http(s)://example.com)"""
    seen, texts = set(), []
    q = collections.deque([(root, 0)])
    domain = urllib.parse.urlparse(root).netloc

    while q and len(seen) < limit_pages:
        url, depth = q.popleft()
        if url in seen or depth > max_depth:
            continue
        seen.add(url)
        try:
            html = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"}).text
            texts.append(clean(html))
            if depth < max_depth:
                soup = BeautifulSoup(html, "html.parser")
                for a in soup.find_all("a", href=True):
                    link = urllib.parse.urljoin(url, a["href"])
                    if urllib.parse.urlparse(link).netloc == domain:
                        q.append((link, depth + 1))
        except Exception as e:
            log.debug("crawl error %s : %s", url, e)
    log.info("Crawl %s → %d pages", domain, len(texts))
    return texts