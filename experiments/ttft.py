import time

from app.graph import graph
from langchain_core.messages import AIMessageChunk


QUESTIONS = [
    "What is TechNova's return policy?",
    "What is the warranty period?",
    "How long does standard shipping take?",
    "Where is my order TN1002?",
    "Can I get a refund?"
]


def has_content(message):
    content = getattr(message, "content", "")

    if isinstance(content, str):
        return bool(content.strip())

    if isinstance(content, list):
        return len(content) > 0

    return bool(content)


ttft_results = []


for i, question in enumerate(QUESTIONS, start=1):

    print("\n" + "=" * 60)
    print(f"Question {i}: {question}")
    print("=" * 60)

    config = {
        "configurable": {
            "thread_id": f"ttft-test-{i}"
        }
    }

    start_time = time.perf_counter()
    first_token_time = None

    try:

        for message, metadata in graph.stream(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": question
                    }
                ]
            },
            config=config,
            stream_mode="messages"
        ):

            # Look for the first non-empty AI message chunk
            if (
                isinstance(message, AIMessageChunk)
                and has_content(message)
            ):

                first_token_time = time.perf_counter() - start_time
                break

        if first_token_time is not None:

            ttft_results.append(first_token_time)

            print(
                f"TTFT: {first_token_time:.3f} seconds"
            )

        else:

            print("TTFT could not be measured.")

    except Exception as e:

        print(f"ERROR: {e}")


if ttft_results:

    average_ttft = sum(ttft_results) / len(ttft_results)

    sorted_results = sorted(ttft_results)

    p50_index = len(sorted_results) // 2
    p50_ttft = sorted_results[p50_index]

    print("\n" + "=" * 60)
    print("TTFT RESULTS")
    print("=" * 60)

    print(
        f"Successful measurements: "
        f"{len(ttft_results)}/{len(QUESTIONS)}"
    )

    print(
        f"Average TTFT: "
        f"{average_ttft:.3f} seconds"
    )

    print(
        f"P50 TTFT: "
        f"{p50_ttft:.3f} seconds"
    )