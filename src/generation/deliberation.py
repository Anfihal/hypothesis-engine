import re
import json
import time
from typing import List, Dict, Optional
from langchain_community.llms import Ollama
from src.llm.client import get_llm_client

class HypothesisDeliberation:
    def __init__(self, use_yandex: bool = False, model_name: str = "llama3.1:8b", timeout: int = 45):
        self.timeout = timeout
        self.use_yandex = use_yandex
        self.model_name = model_name
        # Пытаемся получить Yandex-клиент (если есть ключи)
        self.yandex_client = None
        try:
            self.yandex_client = get_llm_client(use_yandex=True)
        except Exception:
            pass  # Yandex не доступен – игнорируем
    
    def _call_llm(self, prompt: str) -> str:
        """Вызов локальной модели с fallback на Yandex."""
        if not self.use_yandex:
            try:
                llm = Ollama(model=self.model_name, temperature=0.7, timeout=self.timeout)
                response = llm.invoke(prompt)
                return response
            except Exception as e:
                print(f"⚠️ Локальная модель недоступна ({e}). Пробуем Yandex...")
                if self.yandex_client:
                    return self.yandex_client.generate(prompt, temperature=0.7)
                else:
                    raise RuntimeError("Локальная модель не работает, а Yandex не настроен.")
        else:
            if self.yandex_client:
                return self.yandex_client.generate(prompt, temperature=0.7)
            else:
                raise RuntimeError("Yandex выбран, но клиент не настроен (проверьте .env).")
    
    def run(self, kpi: str, constraints: str, context: str, max_rounds: int = 3) -> List[Dict]:
        prompt = f"""
You are a materials scientist. Given the following KPI, constraints, and knowledge context:
KPI: {kpi}
Constraints: {constraints}
Context: {context}

Generate 3 specific, testable hypotheses in JSON format. Each hypothesis must have:
- statement (what to do)
- mechanism (why it works, with reference to specific facts from the context)
- novelty (what makes it new compared to existing knowledge)
- risk (low/medium/high)
- impact (expected effect on KPI in % or qualitative)
- sources (list of specific phrases or facts from the context that support this hypothesis)

Output only a JSON array. Example:
[
  {{
    "statement": "Add 0.3% Nb to alloy X",
    "mechanism": "Nb forms carbides hindering dislocation motion, as mentioned in the context about carbides.",
    "novelty": "First study on this composition at 700°C",
    "risk": "medium",
    "impact": "+15% creep resistance",
    "sources": ["Niobium forms stable carbides", "Two-step aging improves properties"]
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
            hypotheses = [{"statement": response, "mechanism": "", "novelty": "", "risk": "medium", "impact": "", "sources": []}]
        
        for h in hypotheses:
            h['generation_time'] = round(elapsed, 1)
        return hypotheses