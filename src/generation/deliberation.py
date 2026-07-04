import re
import json
import time
from typing import List, Dict, Optional
from langchain_community.llms import Ollama
from src.llm.client import get_llm_client

class HypothesisDeliberation:
    def __init__(
        self,
        use_yandex: bool = False,
        model_name: str = "llama3.1:8b",
        timeout: int = 45,
        enable_search: bool = True,
        search_max_results: int = 3
    ):
        self.timeout = timeout
        self.use_yandex = use_yandex
        self.model_name = model_name
        self.enable_search = enable_search

        # Инициализируем Yandex-клиент только если use_yandex=True
        self.yandex_client = None
        if use_yandex:
            try:
                self.yandex_client = get_llm_client(use_yandex=True)
            except Exception as e:
                raise RuntimeError(f"Не удалось инициализировать Yandex GPT: {e}")

        # Поиск (опционально)
        self.search_manager = None
        if enable_search:
            try:
                from src.search import SearchManager
                self.search_manager = SearchManager(max_results=search_max_results, enable_web=True)
            except ImportError:
                print("⚠️ Модуль поиска не найден. Поиск в интернете отключён.")

    def _call_llm(self, prompt: str) -> str:
        if self.use_yandex:
            if self.yandex_client is None:
                raise RuntimeError("Yandex выбран, но клиент не инициализирован (проверьте .env).")
            return self.yandex_client.generate(prompt, temperature=0.7)
        else:
            try:
                llm = Ollama(model=self.model_name, temperature=0.7, timeout=self.timeout)
                return llm.invoke(prompt)
            except Exception as e:
                raise RuntimeError(f"Локальная модель ({self.model_name}) недоступна. Убедитесь, что Ollama запущен: {e}")

    def run(
        self,
        kpi: str,
        constraints: str,
        context: str,
        language: str = 'en',
        max_rounds: int = 3
    ) -> List[Dict]:
        enhanced_context = context
        if self.enable_search and self.search_manager:
            try:
                search_query = f"{kpi} materials science"
                print(f"🔍 Поиск в интернете по запросу: {search_query}")
                search_results = self.search_manager.search_and_summarize(search_query)
                if search_results and "не найдены" not in search_results:
                    enhanced_context += f"\n\n=== 🌐 Дополнительные источники из интернета ===\n{search_results}"
                    print("✅ Найдены дополнительные источники")
                else:
                    print("ℹ️ Дополнительные источники не найдены")
            except Exception as e:
                print(f"⚠️ Ошибка при поиске: {e}")

        lang_instructions = {
            'en': "Answer in English.",
            'ru': "Ответь на русском языке.",
            'zh': "请用中文回答。"
        }
        lang_prompt = lang_instructions.get(language, "Answer in English.")

        prompt = f"""
You are a materials scientist. {lang_prompt}
Given the following KPI, constraints, and knowledge context (including web search results if provided):
KPI: {kpi}
Constraints: {constraints}
Context: {enhanced_context}

Generate 3 specific, testable hypotheses in JSON format. Each hypothesis must have:
- statement (what to do)
- mechanism (why it works, with reference to specific facts from the context)
- novelty (what makes it new compared to existing knowledge)
- risk (low/medium/high)
- impact (expected effect on KPI in % or qualitative)
- sources (list of specific phrases or facts from the context that support this hypothesis)
- explanation (detailed reasoning for why this hypothesis is plausible)
- recommendation (brief recommendation on how to test or implement this hypothesis)

Output only a JSON array. Example:
[
  {{
    "statement": "Add 0.3% Nb to alloy X",
    "mechanism": "Nb forms carbides hindering dislocation motion, as mentioned in the context about carbides.",
    "novelty": "First study on this composition at 700°C",
    "risk": "medium",
    "impact": "+15% creep resistance",
    "sources": ["Niobium forms stable carbides", "Two-step aging improves properties"],
    "explanation": "Niobium carbides precipitate at grain boundaries, reducing creep by pinning dislocations.",
    "recommendation": "Test with 0.3% Nb and compare with baseline using creep tests at 700°C."
  }}
]
"""
        start_time = time.time()
        response = self._call_llm(prompt)
        elapsed = time.time() - start_time
        print(f"⏱️ Время ответа: {elapsed:.1f} секунд")

        try:
            json_str = re.search(r'\[.*\]', response, re.DOTALL).group()
            hypotheses = json.loads(json_str)
            if not isinstance(hypotheses, list):
                hypotheses = [hypotheses]
        except Exception as e:
            print(f"⚠️ Ошибка парсинга JSON: {e}. Сырой ответ: {response[:200]}...")
            hypotheses = [{
                "statement": response,
                "mechanism": "",
                "novelty": "",
                "risk": "medium",
                "impact": "",
                "sources": [],
                "explanation": "",
                "recommendation": ""
            }]

        for h in hypotheses:
            h['generation_time'] = round(elapsed, 1)
            h.setdefault('sources', [])
            h.setdefault('explanation', '')
            h.setdefault('recommendation', '')

        return hypotheses