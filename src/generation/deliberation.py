import os
import re
import json
import time
from typing import List, Dict, Optional
from dotenv import load_dotenv
from langchain_community.llms import Ollama

from src.llm.client import YandexGPTClient, G4FClient

load_dotenv()


class HypothesisDeliberation:
	def __init__(self, provider: str = "ollama", model: str = None):
		self.provider = provider
		
		if provider == "yandex":
			api_key = os.getenv("YANDEX_API_KEY")
			folder_id = os.getenv("YANDEX_FOLDER_ID")
			self.llm = YandexGPTClient(api_key=api_key, folder_id=folder_id)
		
		elif provider == "ollama":
			model_name = model or "llama3.1:8b"
			self.llm = Ollama(model=model_name)
		
		elif provider == "g4f":
			model_name = model or "gpt-4o-mini"
			self.llm = G4FClient(model=model_name)
		
		else:
			raise ValueError(f"Провайдер {provider} не поддерживается. Выберите yandex, ollama или g4f.")
	
	def generate_hypothesis(self, prompt: str) -> str:
		"""Вызывает LLM для генерации ответа по промпту"""
		response = self.llm.invoke(prompt)
		return response
	
	def run(self, kpi: str, constraints: str, context: str, language: str = 'en', max_rounds: int = 3) -> List[Dict]:
		lang_instructions = {
			'en': "Answer in English.",
			'ru': "Ответь на русском языке.",
			'zh': "请用中文回答。"
		}
		lang_prompt = lang_instructions.get(language, "Answer in English.")
		prompt = f"""
			You are a materials scientist. {lang_prompt}
			Given the following KPI, constraints, and knowledge context:
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
		response = self.generate_hypothesis(prompt)
		elapsed = time.time() - start_time
		print(f"⏱️ Время ответа: {elapsed:.1f} секунд")
		
		try:
			json_str = re.search(r'\[.*\]', response, re.DOTALL).group()
			hypotheses = json.loads(json_str)
			if not isinstance(hypotheses, list):
				hypotheses = [hypotheses]
		except Exception as e:
			print(f"⚠️ Ошибка парсинга JSON: {e}. Сырой ответ: {response[:200]}...")
			hypotheses = [
				{"statement": response, "mechanism": "", "novelty": "", "risk": "medium", "impact": "", "sources": [],
				 "explanation": "", "recommendation": ""}]
		
		for h in hypotheses:
			h['generation_time'] = round(elapsed, 1)
			h.setdefault('sources', [])
			h.setdefault('explanation', '')
			h.setdefault('recommendation', '')
		return hypotheses
