from typing import List, Dict, Optional
from .duckduckgo_search import DuckDuckGoSearch

class SearchManager:
    def __init__(self, max_results: int = 3, enable_web: bool = True):
        self.max_results = max_results
        self.sources = {}
        if enable_web:
            self.sources['duckduckgo'] = DuckDuckGoSearch(max_results * 2)  # больше результатов для веб-поиска
    
    def search_all(self, query: str, sources: Optional[List[str]] = None) -> List[Dict]:
        """Поиск по всем источникам."""
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
        # Сортировка по надёжности (от высокой к низкой)
        all_results.sort(key=lambda x: x.get('reliability', 0), reverse=True)
        return all_results
    
    def _guess_verification(self, item: Dict) -> str:
        """Эвристика для определения проверенности источника."""
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
    
    def search_and_summarize(self, query: str, sources: Optional[List[str]] = None) -> str:
        """Поиск и формирование контекста для LLM."""
        results = self.search_all(query, sources)
        if not results:
            return "Дополнительные источники не найдены."
        parts = []
        for i, item in enumerate(results[:6], 1):
            title = item.get('title', 'Без названия')
            source = item.get('source', '')
            verification = item.get('verification', '')
            abstract = item.get('abstract', '')[:300]
            parts.append(f"[{i}] {title}")
            parts.append(f"   Источник: {source} | {verification}")
            if abstract:
                parts.append(f"   {abstract}...")
            parts.append("")
        return "\n".join(parts)