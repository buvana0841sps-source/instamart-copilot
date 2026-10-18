from pathlib import Path
import re

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from part_d_policy import answer
import mock_llm


BASE_DIR = Path(__file__).resolve().parent
POLICY_DIR = BASE_DIR / "data" / "policies"
EVAL_FILE = BASE_DIR / "data" / "raw" / "policy_eval.csv"
OUTPUT_DIR = BASE_DIR / "outputs"


def find_column(df, possible_names, required=True):
    """Find a column even when its capitalization differs."""
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
            f"Could not find one of {possible_names}. "
            f"CSV columns: {list(df.columns)}"
        )

    return None


def make_chunks(overlap=10):
    """Split policy files into 40-word chunks."""
    if not 0 <= overlap < 40:
        raise ValueError("Overlap must be between 0 and 39.")

    chunks = []
    chunk_size = 40
    step = chunk_size - overlap

    for path in sorted(POLICY_DIR.glob("*.txt")):
        words = path.read_text(encoding="utf-8").split()

        for start in range(0, len(words), step):
            chunk_words = words[start:start + chunk_size]

            if not chunk_words:
                break

            chunks.append({
                "source": path.name,
                "text": " ".join(chunk_words)
            })

            if start + chunk_size >= len(words):
                break

    return chunks


def retrieve_chunks(question, overlap=10, k=3):
    """Retrieve the top-k chunks using TF-IDF and cosine similarity."""
    chunks = make_chunks(overlap)

    if not chunks:
        raise ValueError(f"No policy documents found in {POLICY_DIR}")

    texts = [chunk["text"] for chunk in chunks]
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(texts)
    query_vector = vectorizer.transform([question])

    scores = cosine_similarity(query_vector, matrix).ravel()

    ranked_indices = sorted(
        range(len(chunks)),
        key=lambda i: (-scores[i], i)
    )

    return [chunks[i] for i in ranked_indices[:k]]


def normalize_source(value):
    """Normalize a policy filename for comparison."""
    value = str(value).strip().replace("\\", "/")
    return Path(value).stem.lower()


def contains_expected_source(expected, retrieved):
    """Check whether an expected source appears in the retrieved chunks."""
    if not expected or str(expected).strip().lower() in {
        "none", "nan", "null"
    }:
        return False

    expected_names = {
        normalize_source(part)
        for part in re.split(r"[,;|]", str(expected))
        if part.strip()
    }

    retrieved_names = {
        normalize_source(source) for source in retrieved
    }

    return bool(expected_names & retrieved_names)


def get_answer_text(result):
    """Extract answer text from the policy assistant's result."""
    if isinstance(result, tuple):
        return str(result[0])

    if isinstance(result, dict):
        for key in ("answer", "response", "text"):
            if key in result:
                return str(result[key])

    return str(result)


def looks_like_fallback(response):
    """Check whether the response uses the required fallback prefix."""
    return str(response).strip().lower().startswith("i could not find")


def answer_with_overlap(question, overlap=0, k=3):
    """Generate a fresh answer using chunks with the requested overlap."""
    selected = retrieve_chunks(question, overlap=overlap, k=k)

    context = "\n".join(
        f"[{chunk['source']}] {chunk['text']}"
        for chunk in selected
    )

    messages = [
        {
            "role": "system",
            "content": (
                "POLICY ASSISTANT. Answer only from the supplied "
                "policy context. If the answer cannot be found in "
                "the context, say: I could not find that information "
                "in the policy documents."
            )
        },
        {
            "role": "user",
            "content": f"CONTEXT:\n{context}\nQUESTION:\n{question}"
        }
    ]

    response = mock_llm.chat(messages)
    answer_text = response.choices[0].message.content
    sources = [chunk["source"] for chunk in selected]

    return answer_text, sources


def main():
    if not EVAL_FILE.exists():
        raise FileNotFoundError(
            f"Evaluation file not found: {EVAL_FILE}"
        )

    df = pd.read_csv(EVAL_FILE, keep_default_na=False)

    question_col = find_column(
        df, ["question", "user_question", "query"]
    )

    expected_doc_col = find_column(
        df,
        [
            "expected_doc", "expected_document", "expected_source",
            "expected_policy", "source_file"
        ],
        required=False
    )

    expected_fact_col = find_column(
        df,
        [
            "expected_fact", "expected_answer",
            "expected_key_fact", "fact"
        ],
        required=False
    )

    answerable_col = find_column(
        df, ["answerable", "is_answerable"], required=False
    )

    if expected_doc_col is None or expected_fact_col is None:
        raise ValueError(
            "The evaluation CSV needs expected-document and "
            "expected-fact columns."
        )

    if len(df) != 14:
        print(f"NOTICE: Expected 14 questions; found {len(df)}.")

    results = []

    for index, row in df.iterrows():
        question = str(row[question_col]).strip()
        expected_doc = str(row[expected_doc_col]).strip()
        expected_fact = str(row[expected_fact_col]).strip()

        if answerable_col is not None:
            answerable_value = (
                str(row[answerable_col]).strip().lower()
            )
            answerable = answerable_value in {
                "true", "1", "yes", "y", "answerable"
            }
        else:
            answerable = bool(expected_doc and expected_fact)

        # Normal answer uses the assistant's normal retrieval method.
        normal_answer = get_answer_text(answer(question))

        normal_chunks = retrieve_chunks(
            question, overlap=10, k=3
        )
        normal_sources = [
            chunk["source"] for chunk in normal_chunks
        ]

        # Generate a fresh answer using chunks with zero overlap.
        zero_answer, zero_sources = answer_with_overlap(
            question, overlap=0, k=3
        )

        normal_source_hit = contains_expected_source(
            expected_doc, normal_sources
        )
        zero_source_hit = contains_expected_source(
            expected_doc, zero_sources
        )

        # Required case-sensitive substring checks.
        normal_fact_hit = (
            bool(expected_fact) and expected_fact in normal_answer
        )
        zero_fact_hit = (
            bool(expected_fact) and expected_fact in zero_answer
        )

        normal_fallback_hit = (
            not answerable and looks_like_fallback(normal_answer)
        )
        zero_fallback_hit = (
            not answerable and looks_like_fallback(zero_answer)
        )

        results.append({
            "question": question,
            "answerable": answerable,
            "expected_doc": expected_doc,
            "expected_fact": expected_fact,
            "top3_sources": " | ".join(normal_sources),
            "expected_doc_in_top3": normal_source_hit,
            "expected_fact_in_answer": normal_fact_hit,
            "answer": normal_answer,
            "fallback_correct": normal_fallback_hit,
            "zero_overlap_sources": " | ".join(zero_sources),
            "zero_overlap_expected_doc_in_top3": zero_source_hit,
            "zero_overlap_expected_fact_in_answer": zero_fact_hit,
            "zero_overlap_answer": zero_answer,
            "zero_overlap_fallback_correct": zero_fallback_hit,
        })

        print(f"\nQuestion {index + 1}: {question}")
        print("Normal overlap sources:", normal_sources)
        print("Normal expected document hit:", normal_source_hit)
        print("Normal expected fact hit:", normal_fact_hit)
        print("Zero-overlap sources:", zero_sources)
        print("Zero-overlap expected document hit:", zero_source_hit)
        print("Zero-overlap expected fact hit:", zero_fact_hit)
        print("Zero-overlap answer:", zero_answer)

    results_df = pd.DataFrame(results)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_file = OUTPUT_DIR / "policy_eval_results.csv"
    results_df.to_csv(output_file, index=False)

    answerable_results = results_df[results_df["answerable"]]
    unanswerable_results = results_df[~results_df["answerable"]]

    zero_answerable_fact_failures = answerable_results[
        ~answerable_results["zero_overlap_expected_fact_in_answer"]
    ]

    zero_answerable_source_failures = answerable_results[
        ~answerable_results["zero_overlap_expected_doc_in_top3"]
    ]

    print("\n" + "=" * 60)
    print("POLICY EVALUATION SUMMARY")
    print("=" * 60)
    print("Questions evaluated:", len(results_df))

    print(
        "Normal overlap: expected document in top 3:",
        int(answerable_results["expected_doc_in_top3"].sum()),
        "/", len(answerable_results)
    )
    print(
        "Normal overlap: expected fact in answer:",
        int(answerable_results["expected_fact_in_answer"].sum()),
        "/", len(answerable_results)
    )
    print(
        "Normal overlap: correct unanswerable fallback:",
        int(unanswerable_results["fallback_correct"].sum()),
        "/", len(unanswerable_results)
    )

    print(
        "Zero overlap: expected document in top 3:",
        int(answerable_results[
            "zero_overlap_expected_doc_in_top3"
        ].sum()),
        "/", len(answerable_results)
    )
    print(
        "Zero overlap: expected fact in answer:",
        int(answerable_results[
            "zero_overlap_expected_fact_in_answer"
        ].sum()),
        "/", len(answerable_results)
    )
    print(
        "Zero overlap: correct unanswerable fallback:",
        int(unanswerable_results[
            "zero_overlap_fallback_correct"
        ].sum()),
        "/", len(unanswerable_results)
    )

    print(
        "Answerable questions failing zero-overlap source retrieval:",
        len(zero_answerable_source_failures)
    )
    print(
        "Answerable questions missing the expected fact "
        "with zero overlap:",
        len(zero_answerable_fact_failures)
    )

    if not zero_answerable_fact_failures.empty:
        print("\nAnswerable zero-overlap fact failure examples:")

        for _, item in zero_answerable_fact_failures.iterrows():
            print("Question:", item["question"])
            print("Expected fact:", item["expected_fact"])
            print("Retrieved sources:", item["zero_overlap_sources"])
            print("Zero-overlap answer:", item["zero_overlap_answer"])

    else:
        print(
            "\nNo answerable expected-fact failures were observed "
            "with the current zero-overlap chunks and prompts. "
            "Do not claim a chunk-boundary failure until one is verified."
        )

    print(f"\nDetailed results saved to: {output_file}")


if __name__ == "__main__":
    main()