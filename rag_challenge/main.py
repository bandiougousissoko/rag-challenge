import argparse
import json
import logging
import os
import sys
import requests

from config import (
    PDF_DIR, INDEX_DIR, CACHE_DIR,
    EMAIL, SUBMISSION_NAME,
    SUBMIT_ENDPOINT, CHECK_SUBMISSION_ENDPOINT,
)
from pdf_extractor import extract_all_pdfs
from indexer import FAISSIndex
from gigachat import GigaChatClient
from rag_pipeline import RAGPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def fetch_questions(questions_path: str = None):
    if questions_path and os.path.exists(questions_path):
        logger.info(f"Loading questions from {questions_path}")
        with open(questions_path, encoding="utf-8") as f:
            return json.load(f)

    for candidate in [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "questions.json"),
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "questions.json"),
    ]:
        if os.path.exists(candidate):
            logger.info(f"Found questions at {candidate}")
            with open(candidate, encoding="utf-8") as f:
                return json.load(f)

    logger.error(
        "questions.json not found! Pass it via --questions path/to/questions.json\n"
        "Download from the course materials or from the challenge server UI."
    )
    sys.exit(1)


def build_submission(answers_list: list) -> dict:
    return {
        "team_email": EMAIL,
        "submission_name": SUBMISSION_NAME,
        "answers": answers_list,
    }


def save_submission(submission: dict) -> str:
    fname = f"submission_{SUBMISSION_NAME}.json"
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), fname)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(submission, f, ensure_ascii=False, indent=2)
    logger.info(f"Submission saved → {out_path}")
    return out_path


def check_submission(file_path: str) -> None:
    logger.info(f"Checking submission via {CHECK_SUBMISSION_ENDPOINT} ...")
    try:
        with open(file_path, "rb") as f:
            resp = requests.post(
                CHECK_SUBMISSION_ENDPOINT,
                files={"file": (os.path.basename(file_path), f, "application/json")},
                timeout=30,
            )
        result = resp.json()
        if result.get("status") == "ok":
            logger.info("Check OK: submission is valid")
        else:
            logger.warning(f"Check issues: {result}")
    except Exception as e:
        logger.error(f"Check failed: {e}")


def post_submission(file_path: str) -> None:
    logger.info(f"Submitting to {SUBMIT_ENDPOINT} ...")
    try:
        with open(file_path, "rb") as f:
            resp = requests.post(
                SUBMIT_ENDPOINT,
                files={"file": (os.path.basename(file_path), f, "application/json")},
                timeout=60,
            )
        logger.info(f"Server response: {resp.json()}")
    except Exception as e:
        logger.error(f"Submission POST failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="RAG Challenge solver")
    parser.add_argument("--rebuild",   action="store_true", help="Force rebuild FAISS index")
    parser.add_argument("--submit",    action="store_true", help="POST submission to server")
    parser.add_argument("--questions", default=None,        help="Path to questions JSON file")
    parser.add_argument("--limit",     type=int, default=0, help="Process only first N questions (debug)")
    args = parser.parse_args()

    if not os.path.isdir(PDF_DIR):
        logger.error(f"PDF directory not found: {PDF_DIR}")
        sys.exit(1)

    faiss_index = FAISSIndex()
    loaded = False if args.rebuild else faiss_index.load(INDEX_DIR)

    if not loaded:
        chunks = extract_all_pdfs(PDF_DIR, CACHE_DIR)
        if not chunks:
            logger.error("No chunks extracted — check PDF_DIR")
            sys.exit(1)
        faiss_index.build(chunks)
        faiss_index.save(INDEX_DIR)
    else:
        logger.info(f"Using cached index with {faiss_index.index.ntotal} vectors")

    questions = fetch_questions(args.questions)
    if args.limit:
        questions = questions[:args.limit]
    logger.info(f"Total questions to answer: {len(questions)}")

    llm = GigaChatClient()
    pipeline = RAGPipeline(faiss_index, llm)

    answers = []
    for i, q in enumerate(questions, 1):
        question_text = q.get("question") or q.get("text", "")
        schema        = q.get("schema")  or q.get("kind",  "name")

        logger.info(f"[{i}/{len(questions)}] {schema.upper():8s} | {question_text[:80]}")

        result = pipeline.answer_question(question_text, schema)

        answers.append({
            "question_text": question_text,
            "value":         result["value"],
            "references":    result["references"],
        })

        logger.info(f"  → {result['value']}")

    submission = build_submission(answers)
    out_path = save_submission(submission)

    check_submission(out_path)

    if args.submit:
        post_submission(out_path)

    logger.info("Done!")
    return out_path


if __name__ == "__main__":
    main()
