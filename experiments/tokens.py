from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import ChatNVIDIA

load_dotenv()

MODEL = "nvidia/nemotron-3-super-120b-a12b"

llm = ChatNVIDIA(
    model=MODEL,
    temperature=0
)

questions = [
    "What is TechNova's return policy?",
    "What is the warranty period?",
    "How long does standard shipping take?",
    "Where is my order TN1002?",
    "Can I get a refund?"
]

print("=" * 60)
print(f"MODEL: {MODEL}")
print("=" * 60)

total_input = 0
total_output = 0
total_tokens = 0

for i, question in enumerate(questions, start=1):

    response = llm.invoke(question)

    usage = response.usage_metadata

    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)
    total = usage.get("total_tokens", 0)

    total_input += input_tokens
    total_output += output_tokens
    total_tokens += total

    print(f"\nQuestion {i}: {question}")
    print(f"Input tokens : {input_tokens}")
    print(f"Output tokens: {output_tokens}")
    print(f"Total tokens : {total}")

print("\n" + "=" * 60)
print("TOTALS")
print("=" * 60)

print(f"Total input tokens : {total_input}")
print(f"Total output tokens: {total_output}")
print(f"Total tokens       : {total_tokens}")