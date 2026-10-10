from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import ChatNVIDIA

load_dotenv()

MODEL = "nvidia/nemotron-3-super-120b-a12b"

llm = ChatNVIDIA(
    model=MODEL,
    temperature=0
)

conversation = [
    "What is TechNova?",
    "What is your return policy?",
    "How many days do I have to return a product?",
    "What is the warranty period?",
    "Does the warranty cover manufacturing defects?",
    "How long does standard shipping take?",
    "What about express shipping?",
    "Can I return an unused product?",
    "What happens if my order is delayed?",
    "Can you summarize all the policies we discussed?"
]

history = []

print("=" * 70)
print("10-TURN TOKEN GROWTH EXPERIMENT")
print("=" * 70)

results = []

for turn, question in enumerate(conversation, start=1):

    messages = history + [
        {
            "role": "user",
            "content": question
        }
    ]

    response = llm.invoke(messages)

    usage = response.usage_metadata

    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)
    total_tokens = usage.get("total_tokens", 0)

    results.append(
        (
            turn,
            input_tokens,
            output_tokens,
            total_tokens
        )
    )

    print(f"\nTurn {turn}")
    print(f"User: {question}")
    print(f"Input tokens : {input_tokens}")
    print(f"Output tokens: {output_tokens}")
    print(f"Total tokens : {total_tokens}")

    history.append(
        {
            "role": "user",
            "content": question
        }
    )

    history.append(response)


print("\n" + "=" * 70)
print("TOKEN GROWTH SUMMARY")
print("=" * 70)

print(
    f"{'Turn':<8}"
    f"{'Input':<12}"
    f"{'Output':<12}"
    f"{'Total':<12}"
)

print("-" * 44)

for turn, input_tokens, output_tokens, total_tokens in results:

    print(
        f"{turn:<8}"
        f"{input_tokens:<12}"
        f"{output_tokens:<12}"
        f"{total_tokens:<12}"
    )