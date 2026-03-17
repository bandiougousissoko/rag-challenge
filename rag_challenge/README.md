# RAG Challenge

RAG-система для автоматических ответов на вопросы по годовым отчётам компаний. Читает 100 PDF-файлов, строит векторный индекс и отвечает на вопросы с помощью GigaChat.

## Как это работает

1. `pdf_extractor.py` — извлекает текст из PDF постранично, нарезает на чанки по 512 символов
2. `indexer.py` — превращает чанки в векторы (sentence-transformers) и сохраняет в FAISS-индекс
3. `rag_pipeline.py` — по каждому вопросу ищет похожие чанки и отдаёт их в GigaChat для генерации ответа
4. `main.py` — собирает всё вместе, сохраняет и отправляет submission

## Установка

```bash
pip install -r requirements.txt
```

Для OCR (опционально, нужен только если в PDF сканы):
```bash
brew install tesseract        # macOS
apt install tesseract-ocr     # Ubuntu
```

## Настройка

Скопируйте `.env.example` или создайте `.env` в корне проекта:

```
GIGACHAT_CLIENT_SECRET=ваш_ключ
```

Ключ получить здесь: https://developers.sber.ru/studio → создать проект → Client Secret.

В `config.py` укажите свои данные:

```python
EMAIL           = "ваш@email.com"
SUBMISSION_NAME = "фамилия_v1"
```

## Запуск

```bash
# Задать ключ и запустить
source .env
python -m rag_challenge.main

# Принудительно пересобрать индекс
python -m rag_challenge.main --rebuild

# Отправить submission на сервер
python -m rag_challenge.main --submit

# Указать свой файл с вопросами
python -m rag_challenge.main --questions path/to/questions.json

# Отладка — только первые 5 вопросов
python -m rag_challenge.main --limit 5
```

Готовый файл сохраняется как `submission_<submission_name>.json`.

## Структура проекта

```
rag_challenge/
├── main.py            # точка входа
├── config.py          # настройки (модель, пути, параметры)
├── pdf_extractor.py   # извлечение текста из PDF
├── indexer.py         # построение и поиск по FAISS-индексу
├── gigachat.py        # клиент GigaChat API
├── rag_pipeline.py    # логика RAG: поиск + генерация
├── requirements.txt
└── README.md
```

## Технические параметры

| Компонент | Значение |
|---|---|
| PDF парсинг | PyMuPDF + опциональный OCR (tesseract) |
| Эмбеддинги | `all-MiniLM-L6-v2` (384 dim) |
| Векторный индекс | FAISS IndexFlatIP (cosine similarity) |
| LLM | GigaChat |
| Размер чанка | 512 символов, overlap 80 |
| Top-K | 10 чанков на запрос |

## Формат submission

```json
{
  "team_email": "ваш@email.com",
  "submission_name": "фамилия_v1",
  "answers": [
    {
      "question_text": "Who is the CEO of Apple Inc?",
      "value": "Tim Cook",
      "references": [{"pdf_sha1": "abc123...", "page_index": 5}]
    },
    {
      "question_text": "Did revenue increase in 2023?",
      "value": "yes",
      "references": [{"pdf_sha1": "def456...", "page_index": 12}]
    },
    {
      "question_text": "What was the net profit?",
      "value": "N/A",
      "references": []
    }
  ]
}
```
