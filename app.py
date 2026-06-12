
import streamlit as st

st.set_page_config(
    page_title="Zyro HR Help Desk",
    page_icon="💼",
    layout="wide"
)

st.title("💼 Zyro Dynamics HR Help Desk")

st.markdown(
    "Ask questions about company policies, leave, benefits, probation, travel, and more."
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

question = st.chat_input("Ask an HR question...")

if question:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):
        st.write(question)

    answer = "Backend RAG response goes here"

    with st.chat_message("assistant"):
        st.write(answer)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )
