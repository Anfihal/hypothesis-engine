import re
import json
from typing import List, Dict
from langchain_community.llms import Ollama

class HypothesisDeliberation:
    def __init__(self, use_yandex: bool = False, model_name: str = "llama3.1:8b"):
        self.use_yandex = use_yandex
        if use_yandex:
            # Если вы хотите использовать Yandex, подключите свой клиент
            raise NotImplementedError("Yandex не реализован в этой версии")
        else:
            self.llm = Ollama(model=model_name, temperature=0.7)
    
    def run(self, kpi: str, constraints: str, context: str, max_rounds: int = 3) -> List[Dict]:
        prompt = f"""
You are a materials scientist. Given the following KPI, constraints, and context:
KPI: {kpi}
Constraints: {constraints}
Context: {context}

Generate 3 specific, testable hypotheses in JSON format. Each hypothesis must have:
- statement (what to do)
- mechanism (why it works)
- novelty (what makes it new)
- risk (low/medium/high)
- impact (expected effect on KPI in % or qualitative)

Output only a JSON array. Example:
[
  {{
    "statement": "Add 0.3% Nb to alloy X",
    "mechanism": "Nb forms carbides hindering dislocation motion",
    "novelty": "First study on this composition at 700°C",
    "risk": "medium",
    "impact": "+15% creep resistance"
  }}
]
"""
        response = self.llm.invoke(prompt)
        try:
            json_str = re.search(r'\[.*\]', response, re.DOTALL).group()
            hypotheses = json.loads(json_str)
            if not isinstance(hypotheses, list):
                hypotheses = [hypotheses]
        except:
            hypotheses = [{"statement": response, "mechanism": "", "novelty": "", "risk": "medium", "impact": ""}]
        return hypotheses