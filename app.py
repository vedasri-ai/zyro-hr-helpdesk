import streamlit as st
import os

# ── Page Configuration ──
st.set_page_config(
    page_title="Zyro HR Help Desk",
    page_icon="💼",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ──
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E3A5F;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #E3F2FD;
    }
    .bot-message {
        background-color: #F5F5F5;
    }
    .source-tag {
        font-size: 0.8rem;
        color: #555;
        background-color: #E0E0E0;
        padding: 2px 8px;
        border-radius: 12px;
        margin-right: 5px;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ──
st.markdown('<div class="main-header">💼 Zyro HR Help Desk</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Intelligent HR Assistant powered by RAG</div>', unsafe_allow_html=True)

# ── Sidebar ──
with st.sidebar:
    st.header("About")
    st.write("AI-powered HR assistant for Zyro Dynamics policy documents.")
    st.write("**Documents covered:**")
    st.markdown("""
    - Company Profile
    - Employee Handbook
    - Leave Policy
    - Work From Home Policy
    - Code of Conduct
    - Performance Review Policy
    - Compensation & Benefits
    - IT & Data Security
    - POSH Policy
    - Onboarding & Separation
    - Travel & Expense Policy
    """)
    st.info("Tip: Be specific with your questions for better answers!")

# ── Initialize RAG Pipeline ──
@st.cache_resource
def init_rag():
    """Initialize the RAG pipeline (cached for performance)."""
    from langchain_community.document_loaders import PyPDFDirectoryLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough

    LLM_PROVIDER = "groq"
    LLM_MODEL = "llama3-70b-8192"
    EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
    CORPUS_PATH = "/kaggle/input/zyro-dynamics-hr-corpus/"

    # Load docs
    loader = PyPDFDirectoryLoader(CORPUS_PATH)
    documents = loader.load()

    # Chunk
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512, chunk_overlap=100,
        length_function=len,
        separators=["\\n\\n", "\\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(documents)

    # Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    # Vector store with MMR
    vectorstore = FAISS.from_documents(documents=chunks, embedding=embeddings)
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 20, "lambda_mult": 0.7}
    )

    # LLM
    from langchain_groq import ChatGroq
    llm = ChatGroq(model=LLM_MODEL, temperature=0.1, max_tokens=512)

    # RAG prompt
    RAG_PROMPT = ChatPromptTemplate.from_template("""\\
You are ZyroBot, the HR Help Desk assistant for Zyro Dynamics Pvt. Ltd.
Answer the employee's question using ONLY the provided HR policy context.
Be concise, accurate, and professional.

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
            formatted.append(f"[{i}] Source: {source}\\n{doc.page_content}")
        return "\\n\\n---\\n\\n".join(formatted)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )

    # Guardrail
    GUARDRAIL_PROMPT = ChatPromptTemplate.from_template("""\\
You are a classifier for an HR Help Desk chatbot at Zyro Dynamics Pvt. Ltd.
Determine if a question is related to HR policies.

HR topics include: leave policies, work from home, compensation, benefits,
performance reviews, code of conduct, onboarding, separation, travel expenses,
IT security, sexual harassment prevention, company culture, employee handbook,
probation, salary, promotions, disciplinary actions, and general HR queries.

Respond with ONLY ONE word:
- "IN_SCOPE" if the question is related to HR policies
- "OUT_OF_SCOPE" if the question is unrelated to HR

Question: {question}

Classification:""")

    REFUSAL_MESSAGE = (
        "I am sorry, but I can only answer questions related to HR policies "
        "at Zyro Dynamics Pvt. Ltd. based on our internal policy documents. "
        "Please ask me about leave policies, work from home, compensation, "
        "benefits, performance reviews, code of conduct, onboarding, "
        "separation, travel expenses, IT security, or other HR-related topics."
    )

    guardrail_chain = GUARDRAIL_PROMPT | llm | StrOutputParser()

    return rag_chain, retriever, guardrail_chain, REFUSAL_MESSAGE

# ── Initialize ──
try:
    rag_chain, retriever, guardrail_chain, REFUSAL_MESSAGE = init_rag()
    st.success("✅ HR Policy Bot is ready!")
except Exception as e:
    st.error(f"❌ Error initializing bot: {str(e)}")
    st.stop()

# ── Chat Interface ──
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat history
for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        st.markdown(
            '<div class="chat-message user-message"><strong>👤 You:</strong><br>'
            + msg["content"]
            + "</div>",
            unsafe_allow_html=True,
        )
    else:
        source_html = ""
        if msg.get("sources"):
            tags = [
                '<span class="source-tag">📄 ' + s + "</span>"
                for s in msg["sources"]
            ]
            source_html = "<br><br>Sources: " + " ".join(tags)
        st.markdown(
            '<div class="chat-message bot-message"><strong>🤖 ZyroBot:</strong><br>'
            + msg["content"]
            + source_html
            + "</div>",
            unsafe_allow_html=True,
        )

# ── Input ──
st.divider()
question = st.text_input(
    "Ask your HR question:",
    placeholder="e.g., How many annual leaves do I get?",
    key="question_input",
)

col1, col2 = st.columns([1, 4])
with col1:
    send = st.button("Send")
with col2:
    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

if send and question.strip():
    st.session_state.chat_history.append({"role": "user", "content": question})

    with st.spinner("ZyroBot is thinking..."):
        try:
            classification = guardrail_chain.invoke({"question": question}).strip().upper()

            if "OUT_OF_SCOPE" in classification:
                answer = REFUSAL_MESSAGE
                sources = []
            else:
                answer = rag_chain.invoke(question)
                docs = retriever.invoke(question)
                sources = list(set([
                    d.metadata.get("source", "Unknown").split("/")[-1]
                    for d in docs
                ]))

            st.session_state.chat_history.append({
                "role": "bot",
                "content": answer,
                "sources": sources,
                "classification": classification,
            })

        except Exception as e:
            st.session_state.chat_history.append({
                "role": "bot",
                "content": f"❌ Error: {str(e)}",
                "sources": [],
            })

    st.rerun()