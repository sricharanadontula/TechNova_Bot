from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import ChatNVIDIA

load_dotenv()

MODELS = [
    "openai/gpt-oss-20b",
    "nvidia/nemotron-3-super-120b-a12b"
]

for model_name in MODELS:

    print("\n" + "=" * 60)
    print(f"Testing: {model_name}")
    print("=" * 60)

    try:
        llm = ChatNVIDIA(
            model=model_name,
            temperature=0
        )

        response = llm.invoke(
            "Explain what a customer support chatbot is in one sentence."
        )

        print("SUCCESS")
        print(response.content)

    except Exception as e:
        print("FAILED")
        print(e)