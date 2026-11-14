
import csv
import json
import time
from pathlib import Path

from injecteval.data.questions import evaluation_questions
from injecteval.target.rag import RAGTarget


def run_baseline(rag: RAGTarget, questions: list[dict]):
    results = []

    for i, item in enumerate(questions, start=1):
        question = item["question"]

        print(f"Running question {i}/{len(questions)}")

        start = time.perf_counter()
        result = rag.generate(question)
        latency = time.perf_counter() - start

        retrieved = result["retrieved_documents"]

        results.append({
            "question": question,
            "expected_answer": item["expected"],
            "generated_answer": result["answer"],
            "actual_model": result["model"],
            "defense_enabled": result["defense_enabled"],
            "latency_seconds": round(latency, 3),
            "retrieved_documents": json.dumps([
                {
                    "content": doc["content"],
                    "score": doc["score"],
                    "metadata": doc["metadata"],
                }
                for doc in retrieved
            ], ensure_ascii=False),
            # Fill these in manually after inspecting results.
            "answer_correct": "",
            "relevant_document_retrieved": "",
        })

    output_path = Path("results/baseline.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=results[0].keys(),
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"\nBaseline results saved to {output_path}")

    return results
