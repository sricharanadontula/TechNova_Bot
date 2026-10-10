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


for model_name in MODELS:

    print("\n" + "=" * 70)
    print(f"MODEL: {model_name}")
    print("=" * 70)

    llm = ChatNVIDIA(
        model=model_name,
        temperature=0
    )

    total_input = 0
    total_output = 0
    total_tokens = 0

    successful = 0

    for i, question in enumerate(QUESTIONS, start=1):

        try:
            response = llm.invoke(question)

            usage = response.usage_metadata

            input_tokens = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)
            total = usage.get("total_tokens", 0)

            total_input += input_tokens
            total_output += output_tokens
            total_tokens += total

            successful += 1

            print(
                f"{i:02d}. "
                f"Input={input_tokens:<5} "
                f"Output={output_tokens:<5} "
                f"Total={total:<5}"
            )

        except Exception as e:

            print(f"{i:02d}. FAILED: {e}")

    print("\n" + "-" * 70)
    print("SUMMARY")
    print("-" * 70)

    print(f"Successful requests : {successful}/{len(QUESTIONS)}")
    print(f"Total input tokens  : {total_input}")
    print(f"Total output tokens : {total_output}")
    print(f"Total tokens        : {total_tokens}")

    if successful > 0:
        print(f"Average input/request  : {total_input / successful:.2f}")
        print(f"Average output/request : {total_output / successful:.2f}")
        print(f"Average total/request  : {total_tokens / successful:.2f}")