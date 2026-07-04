import re
import json
import time
import requests
from typing import List, Dict, Optional
from src.llm.client import get_llm_client

class HypothesisDeliberation:
    def __init__(
        self,
        provider: str = "ollama",
        model: str = None,
        timeout: int = 600,
        enable_search: bool = True,
        search_max_results: int = 3,
        warm_up: bool = True
    ):
        self.provider = provider
        self.timeout = timeout
        self.enable_search = enable_search
        self.warm_up = warm_up
        self._warmed_up = False
        self.model_name = model or ("llama3.1:8b" if provider == "ollama" else "gpt-4o-mini")
        self.last_search_results = []

        # Инициализация LLM
        if provider == "yandex":
            self.llm = get_llm_client(use_yandex=True)
        elif provider == "ollama":
            self.llm = None  # будем использовать прямой HTTP
        elif provider == "g4f":
            try:
                from src.llm.client import G4FClient
                self.llm = G4FClient(model=self.model_name)
            except ImportError:
                raise RuntimeError("G4F не установлен. Установите: pip install g4f")
        else:
            raise ValueError(f"Неподдерживаемый провайдер: {provider}")

        # Поисковик
        self.search_manager = None
        if enable_search:
            try:
                from src.search import SearchManager
                self.search_manager = SearchManager(max_results=search_max_results, enable_web=True)
            except ImportError:
                print("⚠️ Модуль поиска не найден. Поиск отключён.")

    def _call_llm(self, prompt: str) -> str:
        if self.provider == "ollama":
            url = "http://localhost:11434/api/generate"
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.7, "num_predict": 2048}
            }
            try:
                if self.warm_up and not self._warmed_up:
                    print("🔄 Прогрев модели...")
                    warm = {"model": self.model_name, "prompt": "Hello", "stream": False}
                    requests.post(url, json=warm, timeout=60).raise_for_status()
                    self._warmed_up = True
                    print("✅ Модель прогрета.")
                resp = requests.post(url, json=payload, timeout=self.timeout)
                resp.raise_for_status()
                return resp.json().get("response", "")
            except requests.exceptions.Timeout:
                raise RuntimeError(f"Таймаут {self.timeout} сек. Попробуйте увеличить таймаут или использовать более лёгкую модель.")
            except Exception as e:
                raise RuntimeError(f"Ошибка Ollama: {e}")
        else:
            # Yandex или G4F
            return self.llm.invoke(prompt)

    def run(self, kpi: str, constraints: str, context: str, language: str = 'en', max_rounds: int = 3) -> List[Dict]:
        enhanced_context = context
        self.last_search_results = []

        if self.enable_search and self.search_manager:
            try:
                search_query = f"{kpi} materials science"
                print(f"🔍 Поиск: {search_query}")
                search_summary = self.search_manager.search_and_summarize(search_query)
                raw = self.search_manager.get_last_results()
                self.last_search_results = raw
                if search_summary and "не найдены" not in search_summary:
                    enhanced_context += f"\n\n=== 🌐 Источники из интернета ===\n{search_summary}"
                    print("✅ Найдены источники")
                else:
                    print("ℹ️ Источников не найдено")
            except Exception as e:
                print(f"⚠️ Ошибка поиска: {e}")

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
- sources (list of specific phrases, facts, or URLs from the context that support this hypothesis)
- explanation (detailed reasoning for why this hypothesis is plausible)
- recommendation (brief recommendation on how to test or implement this hypothesis)

Output only a JSON array.
"""
        start_time = time.time()
        response = self._call_llm(prompt)
        elapsed = time.time() - start_time
        print(f"⏱️ Время ответа: {elapsed:.1f} сек")

        try:
            json_str = re.search(r'\[.*\]', response, re.DOTALL).group()
            hypotheses = json.loads(json_str)
            if not isinstance(hypotheses, list):
                hypotheses = [hypotheses]
        except Exception as e:
            print(f"⚠️ Ошибка парсинга JSON: {e}")
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