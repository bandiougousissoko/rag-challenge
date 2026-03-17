import uuid
import time
import logging
import requests
import urllib3

from config import (
    GIGACHAT_CLIENT_SECRET,
    GIGACHAT_AUTH_URL,
    GIGACHAT_API_URL,
    GIGACHAT_MODEL,
    GIGACHAT_SCOPE,
)

logger = logging.getLogger(__name__)

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class GigaChatClient:
    def __init__(self, client_secret: str = GIGACHAT_CLIENT_SECRET):
        self.client_secret = client_secret
        self._token: str = ""
        self._token_expires: float = 0.0

    def _refresh_token(self) -> None:
        headers = {
            "Content-Type":  "application/x-www-form-urlencoded",
            "Accept":        "application/json",
            "RqUID":         str(uuid.uuid4()),
            "Authorization": f"Basic {self.client_secret}",
        }
        resp = requests.post(
            GIGACHAT_AUTH_URL,
            headers=headers,
            data={"scope": GIGACHAT_SCOPE},
            verify=False,
            timeout=15,
        )
        resp.raise_for_status()
        payload = resp.json()
        self._token = payload["access_token"]
        self._token_expires = payload.get("expires_at", 0) / 1000.0 - 60

    def _get_token(self) -> str:
        if not self._token or time.time() >= self._token_expires:
            self._refresh_token()
        return self._token

    def chat(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.1,
        max_tokens: int = 256,
    ) -> str:
        headers = {
            "Authorization": f"Bearer {self._get_token()}",
            "Content-Type":  "application/json",
            "Accept":        "application/json",
        }
        body = {
            "model": GIGACHAT_MODEL,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_message},
            ],
        }
        for attempt in range(3):
            try:
                resp = requests.post(
                    GIGACHAT_API_URL,
                    headers=headers,
                    json=body,
                    verify=False,
                    timeout=30,
                )
                if resp.status_code == 401:
                    self._token = ""
                    headers["Authorization"] = f"Bearer {self._get_token()}"
                    continue
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"].strip()
            except requests.RequestException as e:
                logger.warning(f"GigaChat attempt {attempt+1} failed: {e}")
                time.sleep(2 ** attempt)

        logger.error("GigaChat: all retries failed, returning N/A")
        return "N/A"
