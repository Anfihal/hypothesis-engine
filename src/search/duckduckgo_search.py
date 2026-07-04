from duckduckgo_search import DDGS
from typing import List, Dict

class DuckDuckGoSearch:
    def __init__(self, max_results: int = 5):
        self.max_results = max_results
        self.ddgs = DDGS()
    
    def search(self, query: str) -> List[Dict]:
        results = []
        try:
            for r in self.ddgs.text(query, max_results=self.max_results):
                results.append({
                    'title': r.get('title', ''),
                    'abstract': r.get('body', '')[:500],
                    'url': r.get('href', ''),
                    'source': 'DuckDuckGo (веб-поиск)',
                    'source_type': 'web',
                    'verification': self._classify(r.get('href', ''), r.get('title', ''), r.get('body', '')),
                    'reliability': self._reliability(r.get('href', ''))
                })
            return results
        except Exception as e:
            print(f"⚠️ Ошибка DuckDuckGo: {e}")
            return []
    
    def _classify(self, url: str, title: str, snippet: str) -> str:
        url_lower = url.lower()
        scientific_domains = ['.edu', '.ac.', 'scholar', 'research', 'sciencedirect', 'springer', 'nature', 'science.org', 'arxiv.org', 'pubmed', 'ncbi', 'ieee', 'acm', 'wiley']
        if any(dom in url_lower for dom in scientific_domains):
            return "🔬 Научный источник (вероятно)"
        elif '.gov' in url_lower:
            return "🏛️ Государственный/официальный источник"
        elif '.org' in url_lower:
            return "🌐 Организационный/некоммерческий источник"
        elif 'news' in url_lower or 'news' in title.lower():
            return "📰 Новостной источник"
        else:
            return "🌍 Общий веб-источник"
    
    def _reliability(self, url: str) -> float:
        high = ['.edu', '.gov', '.ac.', 'nature.com', 'science.org', 'springer.com', 'sciencedirect.com']
        medium = ['.org', 'arxiv.org', 'pubmed', 'wiley.com', 'ieee.org']
        if any(d in url for d in high):
            return 0.9
        elif any(d in url for d in medium):
            return 0.7
        else:
            return 0.5