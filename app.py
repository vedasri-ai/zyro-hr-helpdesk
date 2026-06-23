
import streamlit as st
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq

st.set_page_config(
    page_title="Zyro Dynamics HR Help Desk",
    page_icon="🏢",
    layout="wide"
)

st.title("🏢 Zyro Dynamics HR Help Desk")
st.caption("RAG Powered HR Assistant")

@st.cache_resource
def init_bot():
    CORPUS_PATH = "hr_docs"
    loader = PyPDFDirectoryLoader(CORPUS_PATH)
    documents = loader.load()
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512, chunk_overlap=100,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(documents)
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    
    vectorstore = FAISS.from_documents(documents=chunks, embedding=embeddings)
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 6, "fetch_k": 30, "lambda_mult": 0.5}
    )
    
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.0, max_tokens=1024)
    
    RAG_PROMPT = ChatPromptTemplate.from_template("""You are ZyroBot, the HR Help Desk assistant for Zyro Dynamics Pvt. Ltd.
Answer the employee's question using ONLY the provided HR policy context.
Follow these rules strictly:
1. Be specific and include exact numbers, days, percentages, or amounts from the policy.
2. If the context contains a list, include ALL relevant items.
3. If the context does not have enough information, say "I don't have enough information in the HR policies to answer this."
4. Do NOT make up any information not in the context.
5. Keep the answer clear and well-structured.

Context:
{context}

Question: {question}

Answer:""")
    
    def format_docs(docs):
        if not docs:
            return "No relevant HR policy documents found."
        formatted = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "Unknown").split("/")[-1]
            formatted.append(f"[{i}] Source: {source}\n{doc.page_content}")
        return "\n\n---\n\n".join(formatted)
    
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )
    
    GUARDRAIL_PROMPT = ChatPromptTemplate.from_template("""You are a strict classifier for an HR Help Desk chatbot at Zyro Dynamics Pvt. Ltd.
Classify if the question is about HR policies or company policies.

HR topics include: leave policies (EL, CL, SL, maternity, paternity), work from home, compensation, CTC, salary, benefits, performance reviews, code of conduct, ethics, onboarding, separation, probation, exit, travel expenses, IT security, POSH, sexual harassment, company culture, employee handbook, disciplinary actions, promotions, and employee grievances.

Topics that are NOT HR: coding, programming, sports, weather, politics, math, science, general knowledge, personal life advice, technology questions, and questions about other companies.

Respond with ONLY ONE word, no punctuation:
IN_SCOPE
OUT_OF_SCOPE

Question: {question}

Classification:""")
    
    REFUSAL_MESSAGE = "I apologize, but I can only answer questions related to HR policies at Zyro Dynamics Pvt. Ltd. based on our internal policy documents. Please ask me about topics like leave policies, work from home, compensation, benefits, performance reviews, code of conduct, onboarding, separation, travel expenses, IT security, or other HR-related matters."
    
    guardrail_chain = GUARDRAIL_PROMPT | llm | StrOutputParser()
    
    def ask(question):
        classification = guardrail_chain.invoke({"question": question}).strip().upper()
        if "OUT_OF_SCOPE" in classification:
            return REFUSAL_MESSAGE
        return rag_chain.invoke(question)
    
    return ask

ask = init_bot()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

question = st.chat_input("Ask an HR question...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = ask(question)
        st.write(answer)
    
    st.session_state.messages.append({"role": "assistant", "content": answer})

with st.sidebar:
    st.header("Example Questions")
    st.write("• How many casual leaves are allowed?")
    st.write("• What is the maternity leave policy?")
    st.write("• Can earned leave be encashed?")
    st.write("• What is the retirement age?")
    st.write("• What is the notice period for L7 employees?")
