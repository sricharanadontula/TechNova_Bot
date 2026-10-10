import time
import statistics
from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import ChatNVIDIA

load_dotenv()

MODELS = [
    "openai/gpt-oss-20b",
    "nvidia/nemotron-3-super-120b-a12b"
]

QUESTIONS = [
    "What is TechNova's return policy?",
    "What is the warranty period?",
    "How long does standard shipping take?",
    "Where is my order TN1001?",
    "Where is my order TN1002?",
    "Where is my order TN1003?",
    "Can I return a product after 20 days?",
    "How long does express shipping take?",
    "Does TechNova provide a warranty?",
    "What happens if my order is delayed?",
    "Tell me about TechNova's return policy.",
    "How many days does standard delivery take?",
    "Can I get a refund?",
    "What is the warranty coverage?",
    "Where is order TN1001?",
    "What is the capital of India?",
    "Tell me a joke.",
    "What is Python?",
    "Can you help me with my TechNova order?",
    "Can I return an unused product?"
]


def percentile(values, p):
    values = sorted(values)

    index = (len(values) - 1) * (p / 100)
    lower = int(index)
    upper = min(lower + 1, len(values) - 1)

    if lower == upper:
        return values[lower]

    fraction = index - lower

    return (
        values[lower]
        + (values[upper] - values[lower]) * fraction
    )


for model_name in MODELS:

    print("\n" + "=" * 60)
    print(f"MODEL: {model_name}")
    print("=" * 60)

    llm = ChatNVIDIA(
        model=model_name,
        temperature=0
    )

    latencies = []

    for i, question in enumerate(QUESTIONS, start=1):

        start = time.perf_counter()

        try:
            response = llm.invoke(question)

            elapsed = time.perf_counter() - start
            latencies.append(elapsed)

            print(
                f"{i:02d}. {elapsed:.2f}s | "
                f"{question}"
            )

        except Exception as e:

            elapsed = time.perf_counter() - start

            print(
                f"{i:02d}. FAILED after {elapsed:.2f}s | "
                f"{question}"
            )

            print(f"    Error: {e}")

    if latencies:

        average = statistics.mean(latencies)
        p50 = percentile(latencies, 50)
        p95 = percentile(latencies, 95)
        minimum = min(latencies)
        maximum = max(latencies)

        print("\nRESULTS")
        print("-" * 40)
        print(f"Successful requests : {len(latencies)}/{len(QUESTIONS)}")
        print(f"Average             : {average:.2f}s")
        print(f"P50                 : {p50:.2f}s")
        print(f"P95                 : {p95:.2f}s")
        print(f"Minimum             : {minimum:.2f}s")
        print(f"Maximum             : {maximum:.2f}s")