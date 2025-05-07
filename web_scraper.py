import requests, re, urllib.parse, logging, collections
from bs4 import BeautifulSoup
from typing import List

log = logging.getLogger(__name__)

def clean(html: str) -> str:
    return BeautifulSoup(html, "html.parser").get_text(" ", strip=True)

def crawl_site(root: str, limit_pages: int = 40, max_depth: int = 2) -> List[str]:
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
        except Exception:
            pass
    log.info("[SCRAPER] %d pages crawlées sur %s", len(texts), domain)
    return texts