
import os
import uuid
import logging
import streamlit as st

# Load cloud secrets before importing the chatbot graph.
try:
    for key in [
        "NVIDIA_API_KEY",
        "LANGSMITH_TRACING",
        "LANGSMITH_API_KEY",
        "LANGSMITH_PROJECT",
    ]:
        if key in st.secrets:
            os.environ[key] = str(st.secrets[key])
except (FileNotFoundError, KeyError):
    pass

from app.graph import graph

st.set_page_config(
    page_title="TechNova Support",
    page_icon="💬",
    layout="centered",
)

st.title("💬 TechNova Customer Support")
st.caption("Ask about orders, returns, warranty, and shipping.")

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

with st.sidebar:
    st.subheader("About")
    st.write("AI-powered TechNova customer support.")
    st.write("• Order tracking")
    st.write("• Return policies")
    st.write("• Warranty information")
    st.write("• Shipping information")

    if st.button("New conversation"):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.chat_history = []
        st.rerun()

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("How can I help you today?")

if prompt:
    st.session_state.chat_history.append(
        {"role": "user", "content": prompt}
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Checking your request..."):
            try:
                result = graph.invoke(
                    {"messages": [("human", prompt)]},
                    config={
                        "configurable": {
                            "thread_id": st.session_state.thread_id
                        },
                        "run_name": "technova-web-chat",
                        "tags": ["technova-bot", "web"],
                    },
                )

                answer = result["messages"][-1].content

                if isinstance(answer, list):
                    answer = "\n".join(
                        item.get("text", "")
                        for item in answer
                        if isinstance(item, dict)
                        and item.get("type") == "text"
                    )

                st.markdown(answer)
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": answer}
                )

            except Exception:
                logging.exception("TechNova chatbot request failed")
                st.error(
                    "Sorry, the chatbot encountered an error. "
                    "Please try again later."
                )
