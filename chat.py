from app.graph import graph


config = {
    "configurable": {
        "thread_id": "technova-user-1"
    }
}


print("TechNova Customer Support")
print("Type 'exit' to quit.\n")


while True:

    user_input = input("You: ")

    if user_input.lower() == "exit":
        print("Goodbye!")
        break

    result = graph.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": user_input
                }
            ]
        },
        config=config
    )

    response = result["messages"][-1]

    print("Bot:", response.content)