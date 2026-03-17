import re
import logging
from typing import List, Dict, Any, Tuple

from indexer import FAISSIndex
from gigachat import GigaChatClient
from config import TOP_K

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a precise financial analyst assistant.
Answer the user's question using ONLY the provided document excerpts.

Rules:
- For "number" questions: return ONLY the numeric value (e.g. 42 or 1500000 or 0.35). No units, no commas, no currency signs.
- For "name" questions: return ONLY the person's full name (e.g. "John Smith").
- For "boolean" questions: return ONLY "yes" or "no".
- For "names" questions: return a JSON list of full names (e.g. ["Alice Brown", "Bob Jones"]).
- If the answer is NOT present in the excerpts, return exactly: N/A
- Do NOT guess, invent, or use external knowledge.
- Do NOT include explanation, just the raw answer value.
"""


def _extract_companies(question: str) -> List[str]:
    return re.findall(r'"([^"]+)"', question)


def _build_context(chunks_with_scores: List[Tuple[Any, float]]) -> str:
    parts = []
    for chunk, _ in chunks_with_scores:
        parts.append(
            f"[Source: {chunk.pdf_sha1[:8]}... page {chunk.page_index + 1}]\n{chunk.text}"
        )
    return "\n\n---\n\n".join(parts)


def _parse_answer(raw: str, schema: str):
    raw = raw.strip().strip('"').strip("'")

    if raw.upper() == "N/A" or raw == "":
        return "N/A", True

    if schema == "number":
        clean = re.sub(r"[^\d.\-]", "", raw)
        try:
            val = float(clean)
            return int(val) if val == int(val) else val, False
        except ValueError:
            return "N/A", True

    elif schema == "boolean":
        low = raw.lower()
        if low in ("yes", "true", "1"):
            return "yes", False
        elif low in ("no", "false", "0"):
            return "no", False
        return "N/A", True

    elif schema == "names":
        raw_inner = raw.strip("[]")
        names = [n.strip().strip('"').strip("'") for n in raw_inner.split(",")]
        names = [n for n in names if n]
        if names:
            return names, False
        return "N/A", True

    else:
        return raw, False


class RAGPipeline:
    def __init__(self, index: FAISSIndex, llm: GigaChatClient):
        self.index = index
        self.llm = llm

    def answer_question(self, question: str, schema: str) -> Dict[str, Any]:
        companies = _extract_companies(question)
        search_query = f"{question} {' '.join(companies)}" if companies else question

        if companies:
            chunks_with_scores = self.index.search_with_company_filter(
                search_query, companies[0], top_k=TOP_K
            )
        else:
            chunks_with_scores = self.index.search(search_query, top_k=TOP_K)

        if schema == "boolean" and len(companies) >= 2:
            extra = self.index.search_with_company_filter(
                question, companies[1], top_k=TOP_K // 2
            )
            seen = {(c.pdf_sha1, c.page_index, c.chunk_index) for c, _ in chunks_with_scores}
            for c, s in extra:
                key = (c.pdf_sha1, c.page_index, c.chunk_index)
                if key not in seen:
                    chunks_with_scores.append((c, s))
                    seen.add(key)

        if not chunks_with_scores:
            return {"value": "N/A", "references": []}

        context = _build_context(chunks_with_scores)
        user_msg = f"""Question ({schema}): {question}

Document excerpts:
{context}

Answer:"""

        raw_answer = self.llm.chat(
            system_prompt=SYSTEM_PROMPT,
            user_message=user_msg,
            temperature=0.05,
            max_tokens=128,
        )

        value, is_na = _parse_answer(raw_answer, schema)

        references = []
        if not is_na:
            seen_refs = set()
            for chunk, _ in chunks_with_scores[:3]:
                key = (chunk.pdf_sha1, chunk.page_index)
                if key not in seen_refs:
                    references.append({
                        "pdf_sha1": chunk.pdf_sha1,
                        "page_index": chunk.page_index,
                    })
                    seen_refs.add(key)

        return {"value": value, "references": references}
