
import os

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.prebuilt import ToolNode, tools_condition

from app.tools import get_order_status, get_policy


# Load environment variables from .env when running locally.
# On Streamlit Cloud, the key should be configured in Secrets.
load_dotenv()

# NVIDIA model configuration
MODEL = "nvidia/nemotron-3-super-120b-a12b"

# Read the NVIDIA API key
api_key = os.getenv("NVIDIA_API_KEY")

if not api_key or not api_key.strip():
    raise ValueError(
        "NVIDIA_API_KEY is missing. "
        "Set it in your local .env file or Streamlit Cloud Secrets."
    )

# Initialize the NVIDIA chat model with the API key explicitly.
llm = ChatNVIDIA(
    model=MODEL,
    temperature=0,
    nvidia_api_key=api_key.strip(),
)

# Define the tools available to the chatbot
tools = [
    get_order_status,
    get_policy,
]

# Allow the model to call tools when needed
llm_with_tools = llm.bind_tools(tools)


# Agent node
def agent(state: MessagesState):
    system_message = SystemMessage(
        content="""You are TechNova-Bot, a helpful customer support assistant.

Your responsibilities:
1. Help customers with order status and company policies.
2. Use get_order_status when a customer asks about a specific order.
3. Use get_policy when a customer asks about returns, warranties, or shipping.
4. Do not invent order details or company policies.
5. If the required information is unavailable, explain that clearly.
6. Keep responses clear, friendly, and concise.
7. If a tool returns an error or cannot find an order, explain the issue politely.
"""
    )

    messages = [system_message] + state["messages"]

    response = llm_with_tools.invoke(messages)

    return {"messages": [response]}


# Tool execution node
tool_node = ToolNode(tools)


# Build the LangGraph workflow
graph_builder = StateGraph(MessagesState)

graph_builder.add_node("agent", agent)
graph_builder.add_node("tools", tool_node)

graph_builder.add_edge(START, "agent")

graph_builder.add_conditional_edges(
    "agent",
    tools_condition,
)

graph_builder.add_edge("tools", "agent")


# Add conversation memory
memory = MemorySaver()

graph = graph_builder.compile(
    checkpointer=memory,
)
