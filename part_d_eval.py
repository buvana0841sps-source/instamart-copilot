
from pathlib import Path
import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from part_d_policy import answer


# Project folders
BASE_DIR = Path(__file__).resolve().parent
POLICY_DIR = BASE_DIR / "data" / "policies"
EVAL_FILE = BASE_DIR / "data" / "raw" / "policy_eval.csv"
OUTPUT_DIR = BASE_DIR / "outputs"


def find_column(df, possible_names, required=True):
    """Find a column even if its capitalization differs."""
    normalized = {
        re.sub(r"[^a-z0-9]", "", str(col).lower()): col
        for col in df.columns
    }

    for name in possible_names:
        key = re.sub(r"[^a-z0-9]", "", name.lower())
        if key in normalized:
            return normalized[key]

    if required:
        raise ValueError(
            f"Could not find a column matching {possible_names}. "
            f"CSV columns are: {list(df.columns)}"
        )

    return None


def make_chunks(overlap=10):
    """Split policy files into 40-word chunks."""
    chunks = []
    chunk_size = 40
    step = chunk_size - overlap

    for file_path in sorted(POLICY_DIR.glob("*.txt")):
        words = file_path.read_text(encoding="utf-8").split()

        for start in range(0, len(words), step):
            chunk_words = words[start:start + chunk_size]

            if not chunk_words:
                break

            chunks.append({
                "source": file_path.name,
                "text": " ".join(chunk_words)
            })

            if start + chunk_size >= len(words):
                break

    return chunks


def retrieve_sources(question, overlap=10, k=3):
    """Return the source filenames for the top-k matching chunks."""
    chunks = make_chunks(overlap)

    if not chunks:
        raise ValueError(f"No policy text files found in {POLICY_DIR}")

    texts = [chunk["text"] for chunk in chunks]
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(texts)
    query_vector = vectorizer.transform([question])

    scores = cosine_similarity(query_vector, matrix).ravel()
    best_indices = np.argsort(-scores, kind="stable")[:k]

    return [chunks[i]["source"] for i in best_indices]


def normalize_source(value):
    """Compare filenames regardless of their extensions or paths."""
    value = str(value).strip().replace("\\", "/")
    return Path(value).stem.lower()


def contains_expected_source(expected, retrieved):
    if not expected or pd.isna(expected):
        return False

    expected_names = {
        normalize_source(part)
        for part in re.split(r"[,;|]", str(expected))
        if part.strip()
    }
    retrieved_names = {normalize_source(name) for name in retrieved}

    return bool(expected_names & retrieved_names)


def looks_like_fallback(response):
    """Check whether the response declines to invent an unsupported answer."""
    text = str(response).lower()

    fallback_phrases = [
        "could not find that information",
        "not specified",
        "not mentioned",
        "not stated",
        "not provided",
        "not available",
        "not covered",
        "cannot determine",
        "can't determine",
        "cannot answer",
        "can't answer",
        "do not have enough information",
        "don't have enough information",
        "insufficient information",
        "unable to answer",
        "outside the provided",
        "not in the policy",
        "not included in the policy",
        "i don't know",
        "i do not know",
    ]

    return any(phrase in text for phrase in fallback_phrases)


def get_answer_text(result):
    """Support the answer function's common return formats."""
    if isinstance(result, tuple):
        return str(result[0])

    if isinstance(result, dict):
        for key in ("answer", "response", "text"):
            if key in result:
                return str(result[key])

    return str(result)


def main():
    if not EVAL_FILE.exists():
        raise FileNotFoundError(f"Evaluation file not found: {EVAL_FILE}")

    df = pd.read_csv(EVAL_FILE, keep_default_na=False)

    question_col = find_column(
        df, ["question", "user_question", "query"]
    )
    expected_doc_col = find_column(
        df,
        ["expected_doc", "expected_document", "expected_source",
         "expected_policy", "source_file"],
        required=False
    )
    expected_fact_col = find_column(
        df,
        ["expected_fact", "expected_answer", "expected_key_fact",
         "fact"],
        required=False
    )
    answerable_col = find_column(
        df, ["answerable", "is_answerable"], required=False
    )

    if expected_doc_col is None or expected_fact_col is None:
        raise ValueError(
            "Please check the evaluation CSV headers. "
            "The script needs expected document and expected fact columns."
        )

    if len(df) != 14:
        print(f"NOTICE: Expected 14 evaluation questions; found {len(df)}.")

    results = []

    for index, row in df.iterrows():
        question = str(row[question_col]).strip()
        expected_doc = str(row[expected_doc_col]).strip()
        expected_fact = str(row[expected_fact_col]).strip()

        if answerable_col is not None:
            answerable_value = str(row[answerable_col]).strip().lower()
            answerable = answerable_value in {
                "true", "1", "yes", "y", "answerable"
            }
        else:
            answerable = bool(expected_doc and expected_fact)

        # Evaluate retrieval both with and without chunk overlap.
        top3 = retrieve_sources(question, overlap=10, k=3)
        zero_overlap_top3 = retrieve_sources(
            question, overlap=0, k=3
        )

        # Obtain the policy assistant's answer.
        result = answer(question)
        answer_text = get_answer_text(result)

        source_hit = contains_expected_source(expected_doc, top3)
        zero_overlap_hit = contains_expected_source(
            expected_doc, zero_overlap_top3
        )

        fact_hit = (
            bool(expected_fact)
            and expected_fact.lower() in answer_text.lower()
        )

        fallback_hit = (
            not answerable and looks_like_fallback(answer_text)
        )

        results.append({
            "question": question,
            "answerable": answerable,
            "expected_doc": expected_doc,
            "top3_sources": " | ".join(top3),
            "expected_doc_in_top3": source_hit,
            "zero_overlap_sources": " | ".join(zero_overlap_top3),
            "zero_overlap_expected_doc_in_top3": zero_overlap_hit,
            "expected_fact": expected_fact,
            "expected_fact_in_answer": fact_hit,
            "fallback_correct": fallback_hit,
            "answer": answer_text,
        })

        print(f"\nQuestion {index + 1}: {question}")
        print(f"Top 3 sources: {top3}")
        print(f"Expected document retrieved: {source_hit}")
        print(f"Zero-overlap retrieval: {zero_overlap_hit}")
        print(f"Answer: {answer_text}")

    results_df = pd.DataFrame(results)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_file = OUTPUT_DIR / "policy_eval_results.csv"
    results_df.to_csv(output_file, index=False)

    answerable_results = results_df[results_df["answerable"]]
    unanswerable_results = results_df[~results_df["answerable"]]

    zero_overlap_failures = results_df[
        ~results_df["zero_overlap_expected_doc_in_top3"]
    ]

    print("\n" + "=" * 55)
    print("POLICY EVALUATION SUMMARY")
    print("=" * 55)
    print(f"Questions evaluated: {len(results_df)}")
    print(
        "Answerable questions with correct source in top 3:",
        int(answerable_results["expected_doc_in_top3"].sum()),
        "/", len(answerable_results)
    )
    print(
        "Answerable questions with expected fact in answer:",
        int(answerable_results["expected_fact_in_answer"].sum()),
        "/", len(answerable_results)
    )
    print(
        "Unanswerable questions correctly handled:",
        int(unanswerable_results["fallback_correct"].sum()),
        "/", len(unanswerable_results)
    )
    print(
        "Zero-overlap retrieval failures:",
        len(zero_overlap_failures),
        "/", len(results_df)
    )

    if not zero_overlap_failures.empty:
        print("\nZero-overlap failure examples:")
        for _, item in zero_overlap_failures.iterrows():
            print(f"- {item['question']}")
            print(f"  Expected: {item['expected_doc']}")
            print(f"  Retrieved: {item['zero_overlap_sources']}")

    print(f"\nDetailed results saved to: {output_file}")


if __name__ == "__main__":
    main()
