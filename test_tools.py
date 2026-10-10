from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import ChatNVIDIA

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

messages = [
    {
        "role": "user",
        "content": "Where is my order TN1002?"
    }
]

# Step 1: Ask the LLM
response = llm_with_tools.invoke(messages)

print("LLM requested:")
print(response.tool_calls)

# Step 2: Execute the requested tool
tool_call = response.tool_calls[0]

if tool_call["name"] == "get_order_status":
    tool_result = get_order_status.invoke(tool_call["args"])

elif tool_call["name"] == "get_policy":
    tool_result = get_policy.invoke(tool_call["args"])

else:
    tool_result = "Unknown tool"

print("\nTool result:")
print(tool_result)

# Step 3: Give the tool result back to the LLM
messages.append(response)

messages.append({
    "role": "tool",
    "tool_call_id": tool_call["id"],
    "content": tool_result
})

final_response = llm_with_tools.invoke(messages)

print("\nFinal answer:")
print(final_response.content)