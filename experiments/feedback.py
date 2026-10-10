from dotenv import load_dotenv
from langsmith import Client, traceable
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.messages import SystemMessage, ToolMessage

from app.tools import get_order_status, get_policy

load_dotenv()

client = Client()

MODEL = "nvidia/nemotron-3-super-120b-a12b"

llm = ChatNVIDIA(
    model=MODEL,
    temperature=0
)

tools = [get_order_status, get_policy]
llm_with_tools = llm.bind_tools(tools)


@traceable(
    name="technova-feedback-interaction",
    project_name="technova-bot-sricharana",
    tags=["feedback", "evaluation"]
)
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

    response = llm_with_tools.invoke(messages)

    if response.tool_calls:

        messages.append(response)

        for tool_call in response.tool_calls:

            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_call_id = tool_call["id"]

            if tool_name == "get_order_status":
                tool_result = get_order_status.invoke(tool_args)

            elif tool_name == "get_policy":
                tool_result = get_policy.invoke(tool_args)

            else:
                tool_result = "Tool unavailable."

            messages.append(
                ToolMessage(
                    content=str(tool_result),
                    tool_call_id=tool_call_id
                )
            )

        final_response = llm_with_tools.invoke(messages)

        return final_response.content

    return response.content


feedback_cases = [
    {
        "question": "What is TechNova's return policy?",
        "comment": "The return policy was clear and accurate."
    },
    {
        "question": "How long is the TechNova warranty?",
        "comment": "The warranty information was correct and helpful."
    },
    {
        "question": "What is the status of order TN1001?",
        "comment": "The order status and delivery information were useful."
    },
    {
        "question": "How long does standard shipping take?",
        "comment": "The shipping timeframe was clear."
    },
    {
        "question": "Tell me something unrelated to TechNova.",
        "comment": "The chatbot correctly stayed within TechNova support scope."
    }
]


print("=" * 60)
print("TECHNOVA USER FEEDBACK COLLECTION")
print("=" * 60)


for i, case in enumerate(feedback_cases, start=1):

    print("\n" + "-" * 60)
    print(f"Interaction {i}/5")
    print(f"Question: {case['question']}")
    print()

    try:

        answer = chatbot(case["question"])

        print("Chatbot answer:")
        print(answer)

        # Find the most recent trace created by this interaction
        runs = list(
            client.list_runs(
                project_name="technova-bot-sricharana",
                filter='eq(name, "technova-feedback-interaction")',
                limit=1
            )
        )

        if not runs:
            print("\nError: LangSmith run was not found.")
            continue

        run = runs[0]

        print("\nAdding user feedback...")

        client.create_feedback(
            run_id=run.id,
            key="user_feedback",
            score=1,
            comment=case["comment"]
        )

        print("Feedback recorded successfully.")

    except Exception as e:

        print("\nError:")
        print(type(e).__name__)
        print(str(e))


print("\n" + "=" * 60)
print("FEEDBACK COLLECTION COMPLETED")
print("=" * 60)