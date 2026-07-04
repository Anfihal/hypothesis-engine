from typing import List, Dict

class DuckDuckGoSearch:
    def __init__(self, max_results: int = 5):
        self.max_results = max_results
        self._available = False
        try:
            # Новая библиотека ddgs (заменяет duckduckgo_search)
            from ddgs import DDGS
            self.DDGS = DDGS
            self._available = True
        except ImportError:
            try:
                # Старая версия для обратной совместимости
                from duckduckgo_search import DDGS
                self.DDGS = DDGS
                self._available = True
            except ImportError:
                print("⚠️ Библиотека ddgs (или duckduckgo_search) не установлена. Установите: pip install ddgs")

    def search(self, query: str) -> List[Dict]:
        if not self._available:
            return []
        try:
            ddgs = self.DDGS()
            results = []
            for r in ddgs.text(query, max_results=self.max_results):
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
        scientific = ['.edu', '.ac.', 'scholar', 'research', 'sciencedirect', 'springer', 'nature', 'science.org', 'arxiv.org', 'pubmed', 'ncbi', 'ieee', 'acm', 'wiley']
        if any(dom in url_lower for dom in scientific):
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