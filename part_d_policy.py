
from pathlib import Path
import re

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import mock_llm


# =========================================================
# 1. FILE LOCATIONS
# =========================================================

POLICY_DIR = Path("data/policies")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# 2. SPLIT POLICY DOCUMENTS INTO CHUNKS
# =========================================================

def load_policy_chunks(overlap=10):
    """
    Split each policy into chunks of 40 words.

    With overlap=10, the next chunk starts 30 words
    after the previous chunk's starting position.
    """

    chunks = []
    step = 40 - overlap

    for path in sorted(POLICY_DIR.glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        words = text.split()

        start = 0

        while start < len(words):
            chunk_words = words[start:start + 40]

            chunks.append({
                "source": path.name,
                "text": " ".join(chunk_words)
            })

            if start + 40 >= len(words):
                break

            start += step

    return chunks


chunks = load_policy_chunks(overlap=10)

print("=" * 65)
print("POLICY DOCUMENT CHUNKING")
print("=" * 65)
print("Total chunks:", len(chunks))

for source in sorted({chunk["source"] for chunk in chunks}):
    count = sum(chunk["source"] == source for chunk in chunks)
    print(f"{source}: {count} chunks")

assert len(chunks) == 15, (
    f"Expected 15 chunks, but received {len(chunks)}."
)


# =========================================================
# 3. FIT TF-IDF
# =========================================================

vectorizer = TfidfVectorizer()

chunk_matrix = vectorizer.fit_transform(
    [chunk["text"] for chunk in chunks]
)


# =========================================================
# 4. RETRIEVE THE MOST RELEVANT CHUNKS
# =========================================================

def retrieve(query, k=3):
    """
    Return the k most relevant policy chunks.

    Each result includes source, cosine similarity score,
    and chunk text.
    """

    query_vector = vectorizer.transform([query])

    similarities = cosine_similarity(
        query_vector,
        chunk_matrix
    ).flatten()

    # Stable ordering: score descending, then original chunk order.
    ranked_indices = sorted(
        range(len(chunks)),
        key=lambda i: (-similarities[i], i)
    )

    results = []

    for index in ranked_indices[:k]:
        results.append({
            "source": chunks[index]["source"],
            "score": float(similarities[index]),
            "text": chunks[index]["text"]
        })

    return results


# Check the reference retrieval query.
test_results = retrieve("refund policy late delivery", k=3)

print("\n" + "=" * 65)
print("RETRIEVAL CHECK")
print("=" * 65)

for item in test_results:
    print(
        f"{item['source']} | score={item['score']:.4f}\n"
        f"{item['text']}\n"
    )

assert all(
    item["source"] == "refund_policy.txt"
    for item in test_results
), "The top three results should come from refund_policy.txt."


# =========================================================
# 5. POLICY QUESTION-ANSWERING ASSISTANT
# =========================================================

def answer(question):
    """
    Retrieve three chunks, provide them as context to the
    offline mock model, and return the answer and sources.
    """

    results = retrieve(question, k=3)

    context_lines = [
        f"[{item['source']}] {item['text']}"
        for item in results
    ]

    user_content = (
        "CONTEXT:\n"
        + "\n".join(context_lines)
        + "\nQUESTION:\n"
        + question
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
            "content": user_content
        }
    ]

    response = mock_llm.chat(messages)
    answer_text = response.choices[0].message.content

    sources = list(dict.fromkeys(
        item["source"] for item in results
    ))

    return {
        "answer": answer_text,
        "sources": sources
    }


# =========================================================
# 6. TRY A SAMPLE QUESTION
# =========================================================

if __name__ == "__main__":
    question = (
        "What delivery time makes a completed order "
        "eligible for a full refund?"
    )

    result = answer(question)

    print("\n" + "=" * 65)
    print("SAMPLE POLICY QUESTION")
    print("=" * 65)
    print("Question:", question)
    print("Answer:", result["answer"])
    print("Sources:", ", ".join(result["sources"]))

    print("\nPolicy assistant setup completed.")
