import os
import requests
from dotenv import load_dotenv

load_dotenv()

# === Yandex GPT ===
class YandexGPTClient:
    def __init__(self):
        self.api_key = os.getenv("YANDEX_API_KEY")
        self.folder_id = os.getenv("YANDEX_FOLDER_ID")
        if not self.api_key or not self.folder_id:
            raise ValueError("YANDEX_API_KEY и YANDEX_FOLDER_ID должны быть заданы в .env")
        self.url = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"

    def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 2000) -> str:
        headers = {
            "Authorization": f"Api-Key {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "modelUri": f"gpt://{self.folder_id}/yandexgpt/latest",
            "completionOptions": {"temperature": temperature, "maxTokens": max_tokens},
            "messages": [{"role": "user", "text": prompt}]
        }
        response = requests.post(self.url, headers=headers, json=data)
        response.raise_for_status()
        return response.json()["result"]["alternatives"][0]["message"]["text"]

# === G4F (бесплатные модели) ===
try:
    from g4f.client import Client
    from g4f.Provider import Bing
except ImportError:
    Client = None
    Bing = None

class G4FClient:
    def __init__(self, model: str = "gpt-4o-mini", provider=None):
        if Client is None or Bing is None:
            raise ImportError("G4F не установлен. Установите: pip install g4f")
        # Используем Bing как провайдер без ключей
        self.client = Client(provider=Bing if provider == "bing" else None)
        self.model = model

    def invoke(self, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Ошибка генерации через g4f: {str(e)}"

# === Фабрика для получения LLM-клиента ===
def get_llm_client(use_yandex: bool = False):
    """
    Возвращает клиент для работы с Yandex GPT.
    Для локальной модели (Ollama) возвращает None, так как используется прямой HTTP-запрос.
    """
    if use_yandex:
        return YandexGPTClient()
    else:
        return None  # для локальной модели используется прямой HTTP-запрос