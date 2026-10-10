from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import SystemMessage
from dotenv import load_dotenv

from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.prebuilt import ToolNode, tools_condition

from app.tools import get_order_status, get_policy

load_dotenv()

MODEL = "nvidia/nemotron-3-super-120b-a12b"

llm = ChatNVIDIA(
    model=MODEL,
    temperature=0
)

tools = [
    get_order_status,
    get_policy
]

llm_with_tools = llm.bind_tools(tools)

def agent(state: MessagesState):

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

    messages = [system_message] + state["messages"]

    response = llm_with_tools.invoke(messages)

    return {
        "messages": [response]
    }

tool_node = ToolNode(tools)

graph_builder = StateGraph(MessagesState)

graph_builder.add_node("agent", agent)
graph_builder.add_node("tools", tool_node)

graph_builder.add_edge(START, "agent")

graph_builder.add_conditional_edges(
    "agent",
    tools_condition
)

graph_builder.add_edge("tools", "agent")

memory = MemorySaver()

graph = graph_builder.compile(
    checkpointer=memory
)

