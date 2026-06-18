
import streamlit as st

st.set_page_config(
    page_title="Zyro Dynamics HR Help Desk",
    page_icon="🏢",
    layout="wide"
)

st.title("🏢 Zyro Dynamics HR Help Desk")
st.caption("RAG Powered HR Assistant")

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

    answer = ask(question)

    with st.chat_message("assistant"):
        st.write(answer)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )

with st.sidebar:
    st.header("Example Questions")

    st.write("• How many casual leaves are allowed?")
    st.write("• What is the maternity leave policy?")
    st.write("• Can earned leave be encashed?")
    st.write("• What is the retirement age?")
    st.write("• What is the notice period for L7 employees?")
