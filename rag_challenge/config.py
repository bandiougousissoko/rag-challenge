import os

GIGACHAT_CLIENT_SECRET = os.environ["GIGACHAT_CLIENT_SECRET"]
GIGACHAT_AUTH_URL = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
GIGACHAT_API_URL = "https://gigachat.devices.sberbank.ru/api/v1/chat/completions"
GIGACHAT_MODEL = "GigaChat"
GIGACHAT_SCOPE = "GIGACHAT_API_PERS"

CHALLENGE_BASE_URL = "http://5.35.3.130:800"
SUBMIT_ENDPOINT = f"{CHALLENGE_BASE_URL}/submit"
CHECK_SUBMISSION_ENDPOINT = f"{CHALLENGE_BASE_URL}/check-submission"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PDF_DIR = os.path.join(BASE_DIR, "pdfs")
INDEX_DIR = os.path.join(BASE_DIR, "rag_challenge", "index")
CACHE_DIR = os.path.join(BASE_DIR, "rag_challenge", "cache")

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 80
TOP_K = 10

EMAIL = "lazarrosissoko24@gmail.com"
SUBMISSION_NAME = "sissoko_v3"
