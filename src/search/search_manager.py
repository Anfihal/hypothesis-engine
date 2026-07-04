from typing import List, Dict, Optional
from .duckduckgo_search import DuckDuckGoSearch

class SearchManager:
    def __init__(self, max_results: int = 3, enable_web: bool = True):
        self.max_results = max_results
        self.sources = {}
        self._last_results = []
        if enable_web:
            self.sources['duckduckgo'] = DuckDuckGoSearch(max_results * 2)

    def search_all(self, query: str, sources: Optional[List[str]] = None) -> List[Dict]:
        if sources is None:
            sources = list(self.sources.keys())
        all_results = []
        for name in sources:
            if name in self.sources:
                try:
                    items = self.sources[name].search(query)
                    for item in items:
                        item['source_name'] = name
                        if 'verification' not in item:
                            item['verification'] = self._guess_verification(item)
                    all_results.extend(items)
                except Exception as e:
                    print(f"⚠️ Ошибка {name}: {e}")
        all_results.sort(key=lambda x: x.get('reliability', 0), reverse=True)
        self._last_results = all_results
        return all_results

    def get_last_results(self) -> List[Dict]:
        """Возвращает результаты последнего поиска."""
        return self._last_results

    def search_and_summarize(self, query: str, sources: Optional[List[str]] = None) -> str:
        results = self.search_all(query, sources)
        if not results:
            return "Дополнительные источники не найдены."
        parts = []
        for i, item in enumerate(results[:6], 1):
            title = item.get('title', 'Без названия')
            url = item.get('url', '')
            verification = item.get('verification', '')
            abstract = item.get('abstract', '')[:300]
            parts.append(f"[{i}] {title}")
            if url:
                parts.append(f"   URL: {url}")
            parts.append(f"   Источник: {item.get('source', '')} | {verification}")
            if abstract:
                parts.append(f"   Аннотация: {abstract}...")
            parts.append("")
        return "\n".join(parts)

    def _guess_verification(self, item: Dict) -> str:
        source = item.get('source', '').lower()
        if 'arxiv' in source:
            return "📄 Препринт (arXiv)"
        elif 'pubmed' in source or 'semantic scholar' in source:
            return "✅ Рецензируемая база данных"
        elif 'nature' in source or 'science' in source:
            return "✅ Высокорейтинговый журнал"
        elif 'wikipedia' in source:
            return "📚 Энциклопедия (проверяемая)"
        elif '.edu' in item.get('url', '').lower():
            return "🎓 Образовательное учреждение"
        elif '.gov' in item.get('url', '').lower():
            return "🏛️ Государственный источник"
        else:
            return "🌍 Веб-источник (требуется проверка)"