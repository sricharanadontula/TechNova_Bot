from dotenv import load_dotenv
from langsmith import Client
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.messages import SystemMessage, ToolMessage
import re

from app.tools import get_order_status, get_policy

load_dotenv()

client = Client()

DATASET_NAME = "TechNova Customer Support Evaluation"

CHATBOT_MODEL = "openai/gpt-oss-20b"
JUDGE_MODEL = "openai/gpt-oss-20b"


# --------------------------------------------------
# FIND DATASET
# --------------------------------------------------

datasets = list(
    client.list_datasets(
        dataset_name=DATASET_NAME
    )
)

if not datasets:
    raise ValueError(
        f"Dataset '{DATASET_NAME}' was not found."
    )

dataset = datasets[0]


# --------------------------------------------------
# CHATBOT MODEL
# --------------------------------------------------

llm = ChatNVIDIA(
    model=CHATBOT_MODEL,
    temperature=0
)

tools = [
    get_order_status,
    get_policy
]

llm_with_tools = llm.bind_tools(tools)


# --------------------------------------------------
# JUDGE MODEL
# --------------------------------------------------

judge_llm = ChatNVIDIA(
    model=JUDGE_MODEL,
    temperature=0
)


# --------------------------------------------------
# CHATBOT
# --------------------------------------------------

def chatbot(question):

    system_message = SystemMessage(
        content="""
You are TechNova's customer support assistant.

Your responsibilities:
- Help customers with orders and company policies.
- Use get_order_status when the customer asks about an order.
- Use get_policy for returns, warranty, or shipping questions.
- Never invent order information or company policies.
- If the required information is unavailable, clearly say so.
- Be concise, friendly, and professional.
- If a customer asks something unrelated to TechNova support,
  politely explain that you can only help with TechNova-related questions.
"""
    )

    messages = [
        system_message,
        ("human", question)
    ]

    # First LLM call
    response = llm_with_tools.invoke(messages)

    # --------------------------------------------------
    # TOOL CALL
    # --------------------------------------------------

    if response.tool_calls:

        messages.append(response)

        for tool_call in response.tool_calls:

            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_call_id = tool_call["id"]

            if tool_name == "get_order_status":

                tool_result = get_order_status.invoke(
                    tool_args
                )

            elif tool_name == "get_policy":

                tool_result = get_policy.invoke(
                    tool_args
                )

            else:

                tool_result = (
                    "The requested tool is unavailable."
                )

            # IMPORTANT:
            # Include the tool_call_id.
            messages.append(
                ToolMessage(
                    content=str(tool_result),
                    tool_call_id=tool_call_id
                )
            )

        # Second LLM call
        final_response = llm_with_tools.invoke(
            messages
        )

        return final_response.content

    return response.content


# --------------------------------------------------
# REFERENCE ANSWERS
# --------------------------------------------------

def get_reference_answer(question):

    references = {

        "Tell me something unrelated to TechNova.": (
            "The chatbot should politely refuse unrelated requests "
            "and explain that it can only help with TechNova-related "
            "orders, returns, warranty, and shipping."
        ),

        "Does TechNova provide a one-year warranty?": (
            "Yes. TechNova products include a one-year limited "
            "warranty covering manufacturing defects."
        ),

        "What should I do if my order ID is TN999999?": (
            "No order was found with order ID TN999999. "
            "The chatbot should not invent an order status and "
            "should suggest checking the order ID or contacting support."
        ),

        "Can I return a product after 40 days?": (
            "No. TechNova accepts returns within 30 days of delivery. "
            "The product must be unused and in its original packaging."
        ),

        "Can I return a product after 20 days?": (
            "Yes. A return after 20 days is within TechNova's "
            "30-day return window, provided the product is unused "
            "and in its original packaging."
        ),

        "What is the status of order TN1003?": (
            "Order TN1003 is Delivered. "
            "The item is a Bluetooth Speaker and the expected "
            "delivery date is October 3, 2026."
        ),

        "What is the status of order TN1002?": (
            "Order TN1002 is Shipped. "
            "The item is a Smart Watch and the expected "
            "delivery date is October 8, 2026."
        ),

        "What is the status of order TN1001?": (
            "Order TN1001 is Processing. "
            "The item is Wireless Headphones and the expected "
            "delivery date is October 10, 2026."
        ),

        "How long does express shipping take?": (
            "Express shipping usually takes 1 to 2 business days."
        ),

        "How long does standard shipping take?": (
            "Standard shipping usually takes 3 to 5 business days."
        ),

        "How long is the TechNova warranty?": (
            "TechNova products include a one-year limited warranty "
            "covering manufacturing defects."
        ),

        "What is TechNova's return policy?": (
            "TechNova accepts returns within 30 days of delivery. "
            "The product must be unused and in its original packaging."
        ),
    }

    return references.get(
        question,
        "No reference answer available."
    )


# --------------------------------------------------
# JUDGE
# --------------------------------------------------

def judge_answer(question, answer):

    reference_answer = get_reference_answer(question)

    prompt = f"""
You are the quality judge for a controlled TechNova
customer-support chatbot benchmark.

TechNova is a fictional company created specifically
for this benchmark.

The REFERENCE ANSWER is the ground truth.

Do not require external verification.
Do not use outside knowledge.

Information matching the reference answer is correct
and grounded.

A polite refusal of an unrelated request is correct
when the reference answer requires a refusal.

Evaluate:

1. Correctness
2. Groundedness
3. Relevance
4. Helpfulness

Scoring:

5 = Excellent
4 = Good
3 = Acceptable
2 = Poor
1 = Completely incorrect

REFERENCE ANSWER:
{reference_answer}

CUSTOMER QUESTION:
{question}

CHATBOT ANSWER:
{answer}

Return exactly:

Score: <1-5>
Reason: <short explanation>
"""

    response = judge_llm.invoke(prompt)

    text = response.content

    match = re.search(
        r"Score:\s*([1-5])",
        text
    )

    if match:
        score = int(match.group(1))
    else:
        score = None

    return score, text


# --------------------------------------------------
# GET DATASET EXAMPLES
# --------------------------------------------------

examples = list(
    client.list_examples(
        dataset_id=dataset.id
    )
)


# --------------------------------------------------
# HEADER
# --------------------------------------------------

print("=" * 60)
print("TECHNOVA SECOND-MODEL EVALUATION")
print("=" * 60)

print(f"Dataset: {DATASET_NAME}")
print(f"Chatbot model: {CHATBOT_MODEL}")
print(f"Judge model: {JUDGE_MODEL}")
print(f"Questions: {len(examples)}")


# --------------------------------------------------
# RUN EVALUATION
# --------------------------------------------------

results = []

for i, example in enumerate(
    examples,
    start=1
):

    question = example.inputs["question"]

    print("\n" + "-" * 60)
    print(f"Question {i}/{len(examples)}")
    print(question)

    try:

        answer = chatbot(question)

        score, evaluation = judge_answer(
            question,
            answer
        )

        results.append(
            {
                "question": question,
                "answer": answer,
                "score": score
            }
        )

        print("\nChatbot answer:")
        print(answer)

        print("\nJudge:")
        print(evaluation)

    except Exception as e:

        print("\nEvaluation failed:")
        print(type(e).__name__)
        print(str(e))

        results.append(
            {
                "question": question,
                "answer": "ERROR",
                "score": None
            }
        )


# --------------------------------------------------
# SUMMARY
# --------------------------------------------------

valid_scores = [
    result["score"]
    for result in results
    if result["score"] is not None
]


print("\n" + "=" * 60)
print("SECOND-MODEL EVALUATION SUMMARY")
print("=" * 60)

print(f"Total questions: {len(results)}")
print(f"Valid scores: {len(valid_scores)}")


if valid_scores:

    average_score = (
        sum(valid_scores) / len(valid_scores)
    )

    print(
        f"Average judge score: "
        f"{average_score:.2f}/5"
    )

    print("\nScore distribution:")

    for score in range(1, 6):

        count = valid_scores.count(score)

        print(
            f"Score {score}: "
            f"{count} question(s)"
        )


print("\n" + "=" * 60)
print("SECOND-MODEL EVALUATION COMPLETE")
print("=" * 60)