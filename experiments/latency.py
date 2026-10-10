import time
import statistics

from app.graph import graph


QUESTIONS = [
    "What is your return policy?",
    "What is your warranty policy?",
    "How long does standard shipping take?",
    "Where is my order TN1001?",
    "Where is my order TN1002?",
    "Where is my order TN1003?",
    "Tell me about returns.",
    "Tell me about shipping.",
    "What does the warranty cover?",
    "Can you check order TN1002?",
    "What is the status of TN1001?",
    "When will TN1003 arrive?",
    "What happens if I want to return an item?",
    "How long does express shipping take?",
    "Is there a warranty?",
    "Who is the president of the United States?",
    "Tell me a joke.",
    "What is Python?",
    "I need help with my TechNova order.",
    "Can you help me with a return?"
]


def run_question(question, index):
    config = {
        "configurable": {
            "thread_id": f"latency-test-{index}"
        }
    }

    start = time.perf_counter()

    result = graph.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": question
                }
            ]
        },
        config=config
    )

    end = time.perf_counter()

    latency = end - start

    return latency, result


latencies = []


print("Starting latency benchmark...\n")


for i, question in enumerate(QUESTIONS, start=1):

    latency, result = run_question(question, i)

    latencies.append(latency)

    print(
        f"{i:02d}. {latency:.2f}s | {question}"
    )


average = statistics.mean(latencies)

p50 = statistics.median(latencies)

p95 = statistics.quantiles(
    latencies,
    n=100
)[94]


print("\n" + "=" * 50)
print("LATENCY RESULTS")
print("=" * 50)

print(f"Average: {average:.2f}s")
print(f"P50:     {p50:.2f}s")
print(f"P95:     {p95:.2f}s")
print(f"Min:     {min(latencies):.2f}s")
print(f"Max:     {max(latencies):.2f}s")