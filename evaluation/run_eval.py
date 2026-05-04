import csv
import os

from evaluation.evaluator import Evaluator

# -------------------------
# Run evaluation first
# -------------------------
evaluator = Evaluator()
evaluator.run()

input_path = os.path.join("evaluation", "test_results.csv")

total = 0
disease_correct = 0
refusal_correct = 0
answer_count = 0

failures = []

with open(input_path, mode="r", encoding="utf-8") as file:
    reader = csv.DictReader(file)

    for row in reader:
        total += 1

        expected = row["expected_disease"].lower().strip()
        predicted = row["predicted_disease"].lower().strip()
        query_type = row["type"].strip()
        is_refusal = row["is_refusal"] == "True"

        # -------------------------
        # Disease accuracy
        # -------------------------
        if expected and predicted == expected:
            disease_correct += 1

        # -------------------------
        # Refusal accuracy
        # -------------------------
        if query_type == "refusal" and is_refusal:
            refusal_correct += 1
        elif query_type == "normal" and not is_refusal:
            refusal_correct += 1

        # -------------------------
        # Answer rate
        # -------------------------
        if query_type == "normal" and not is_refusal:
            answer_count += 1

        # -------------------------
        # Fail tracking
        # -------------------------
        if expected and predicted != expected:
            failures.append((row["query"], predicted, expected))

# -------------------------
# Results
# -------------------------

print("\n--- EVALUATION ---\n")

print(f"Total queries: {total}")

if total > 0:
    print(f"Disease Accuracy: {disease_correct / total:.2f}")
    print(f"Refusal Accuracy: {refusal_correct / total:.2f}")

normal_queries = sum(1 for _ in open(input_path)) - 1
print(f"Answer Rate (normal queries): {answer_count}")

print("\n--- Failures (Retrieval mismatch) ---")
for f in failures:
    print(f"Query: {f[0]}")
    print(f"Predicted: {f[1]} | Expected: {f[2]}\n")