# Hypothesis Engine

Прототип мультиагентной системы для генерации и ранжирования научных гипотез в материаловедении.

## Установка

1. Создайте виртуальное окружение: `python -m venv venv`
2. Активируйте: `.\venv\Scripts\Activate.ps1` (Windows)
3. Установите зависимости: `pip install -r requirements.txt`
4. Скачайте модель SciSpacy: `python -m spacy download en_core_sci_lg`
5. Настройте `.env` с ключами Yandex API (опционально)

## Запуск

```bash
python src/main.py