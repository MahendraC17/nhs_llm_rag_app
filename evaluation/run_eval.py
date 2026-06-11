# --------------------------------------------------------------------------------
# Evaluation Runner
# Computing aggregate metrics from evaluation output
# --------------------------------------------------------------------------------

import csv
import os

from evaluation.evaluator import Evaluator

# Run evaluation
evaluator = Evaluator()
evaluator.run()

input_path = os.path.join("evaluation", "test_results.csv")

total = 0
normal_correct = 0
refusal_correct = 0
ambiguous_correct = 0
normal_total = 0
refusal_total = 0
ambiguous_total = 0

failures = []

def is_ambiguous_response(response):
    response = response.lower()

    return (
        "not confident enough" in response
        or "symptoms are too broad" in response
        or "don't clearly match one condition" in response)


with open(input_path, mode="r", encoding="utf-8") as file:
    reader = csv.DictReader(file)

    for row in reader:
        total += 1
        query = row["query"]
        query_type = row["type"].strip()
        response = row["response"]
        is_refusal = row["is_refusal"] == "True"
        final_status = row["final_status"]

        if query_type == "normal":
            normal_total += 1

            if row["final_status"] == "SUCCESS":
                normal_correct += 1
            else:
                failures.append(
                    (query, final_status))

        elif query_type == "refusal":
            refusal_total += 1
            final_status = row["final_status"]

            if final_status == "NON_MEDICAL":
                refusal_correct += 1
            else:
                failures.append(
                    (query, "REFUSAL_QUERY_ANSWERED"))

        elif query_type == "ambiguous":
            ambiguous_total += 1

            if row["final_status"] == "AMBIGUOUS_QUERY":
                ambiguous_correct += 1
            else:
                failures.append(
                    (query, "AMBIGUOUS_QUERY_ANSWERED"))

print("\n--- EVALUATION RUN ---\n")

print(f"Total queries: {total}")

if normal_total:
    print(f"Normal Query Accuracy: "f"{normal_correct / normal_total:.2f}")

if refusal_total:
    print(f"Refusal Accuracy: "f"{refusal_correct / refusal_total:.2f}")

if ambiguous_total:
    print(f"Ambiguous Accuracy: "f"{ambiguous_correct / ambiguous_total:.2f}")

print(f"Successful Answers: {normal_correct}")

print("\n--- Failures ---")

for query, reason in failures:
    print(f"{reason}: {query}")