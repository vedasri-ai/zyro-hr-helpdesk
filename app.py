import os
import streamlit as st
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq

GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY", ""))
CORPUS_PATH  = "hr_docs/"

st.set_page_config(
    page_title="Zyro Dynamics HR Help Desk",
    page_icon="🏢",
    layout="centered",
)

st.title("Zyro Dynamics — HR Help Desk")
st.caption("Powered by RAG + Groq | Ask anything about HR policies")

@st.cache_resource(show_spinner="Loading HR policy documents...")
def load_pipeline():
    loader = PyPDFDirectoryLoader(CORPUS_PATH)
    docs = loader.load()

    for doc in docs:
        src = doc.metadata.get("source", "")
        doc.metadata["filename"] = os.path.basename(src)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)

    emb = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        encode_kwargs={"normalize_embeddings": True},
    )

    vs = FAISS.from_documents(chunks, emb)
    retriever = vs.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 20, "lambda_mult": 0.7},
    )

    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model="llama-3.3-70b-versatile",
        temperature=0.1,
        max_tokens=512,
    )

    rag_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are ZyroHR, the official HR Help Desk assistant for Zyro Dynamics Pvt. Ltd. "
         "IMPORTANT: Acrux Dynamics and Zyro Dynamics are the SAME company. "
         "The documents may use either name — treat them as identical. "
         "Never say information is unavailable just because the question says Acrux Dynamics.\n\n"
         "Answer employee questions using ONLY the provided HR policy context. "
         "Rules:\n"
         "- Be concise and factual.\n"
         "- Always mention which document your answer comes from.\n"
         "- For compensation questions: state exact CTC range and bonus target percentage for the grade.\n"
         "- For insurance questions: cover ALL three types:\n"
         "  1) Group Medical Insurance: Rs. 5,00,000 per year floater policy covering employee, spouse, and up to two dependent children. All premiums fully paid by the company.\n"
         "  2) Personal Accident Insurance: coverage of 5 times the employee's annual CTC.\n"
         "  3) Term Life Insurance: coverage of 3 times the annual CTC for all permanent employees.\n"
         "- For leave questions: state exact entitlement days, eligibility criteria, carry forward limit, and encashment rules.\n"
         "- For WFH questions: list all four arrangement types (Hybrid, Full Remote, Ad-hoc, Emergency) with eligibility grades and max days per week.\n"
         "- For separation questions: state notice period by grade, F&F processing timeline (within 30 days), and all components included.\n"
         "- For ESOP questions: state eligibility grade (L5 and above), full vesting schedule (25% Year 1, 25% Year 2, 50% Year 4 on a 1-year cliff), and clearly state the number of options granted is not specified in the policy.\n"
         "- For payroll questions: salary credited by 7th of following month, payroll cut-off is 24th of each month.\n"
         "- For performance review questions: cover the full APR timeline, all 5 rating labels with descriptions, PIP criteria (rating 1 or 2 in two consecutive cycles), and promotion eligibility requirements.\n"
         "- For maternity leave: state 26 weeks for first two births, 80 days minimum service requirement, and 8 weeks pre-natal leave allowance.\n"
         "- Only say you cannot find the answer if the topic is genuinely not covered anywhere in the context.\n"
         "- Never fabricate numbers, dates, or policy details not present in the context."),
        ("human", "HR Policy Context:\n{context}\n\nEmployee Question: {question}\n\nAnswer:"),
    ])

    oos_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a query classifier for the Zyro Dynamics HR Help Desk.\n"
         "Classify the question as HR-RELATED or OUT-OF-SCOPE.\n\n"
         "HR-RELATED includes anything about: leave, salary, CTC, payroll, bonus, PF, "
         "gratuity, insurance, ESOP, attendance, working hours, WFH, remote work, "
         "performance review, appraisal, PIP, promotion, demotion, termination, firing, "
         "resignation, notice period, onboarding, probation, separation, F&F settlement, "
         "exit, retirement, travel, expense, reimbursement, code of conduct, POSH, "
         "harassment, IT policy, data security, company profile, grade levels, benefits, "
         "perks, L&D budget, wellness, Zyro Dynamics policies, Acrux Dynamics policies.\n\n"
         "IMPORTANT: Acrux Dynamics and Zyro Dynamics refer to the same company. "
         "Any question about Acrux Dynamics HR policies is HR-RELATED.\n\n"
         "OUT-OF-SCOPE: cooking, sports scores, movies, general coding, weather, "
         "stock markets, financial performance, product comparisons with other companies, "
         "recruitment/hiring process, anything unrelated to internal HR policies.\n\n"
         "Reply with ONE word only: HR-RELATED or OUT-OF-SCOPE."),
        ("human", "{question}"),
    ])

    def format_docs(docs):
        return "\n\n".join(
            f"[{d.metadata.get('filename', 'HR Policy')} - Page {d.metadata.get('page', 0) + 1}]\n"
            f"{d.page_content.strip()}"
            for d in docs
        )

    def ask(question):
        label = (oos_prompt | llm | StrOutputParser()).invoke({"question": question}).strip().upper()
        if "HR-RELATED" not in label:
            return {
                "answer": "I can only answer questions related to Zyro Dynamics HR policies "
                          "such as leave, salary, benefits, attendance, conduct, and separation. "
                          "Your question is outside my scope. Please contact the relevant department directly.",
                "sources": [],
                "is_hr": False,
            }

        rdocs = retriever.invoke(question)
        context = format_docs(rdocs)

        chain = (
            {"context": lambda _: context, "question": RunnablePassthrough()}
            | rag_prompt
            | llm
            | StrOutputParser()
        )
        answer = chain.invoke(question)
        sources = sorted({d.metadata.get("filename", "HR Policy") for d in rdocs})

        return {"answer": answer, "sources": sources, "is_hr": True}

    return ask

if "messages" not in st.session_state:
    st.session_state.messages = []

if not GROQ_API_KEY:
    st.error("GROQ_API_KEY not found. Add it in Streamlit Secrets.")
    st.stop()

ask = load_pipeline()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            st.caption("Sources: " + " | ".join(msg["sources"]))

if not st.session_state.messages:
    st.markdown("**Suggested questions:**")
    suggestions = [
        "How many earned leaves do I get per year?",
        "What is the WFH policy for L3 employees?",
        "What is the notice period if I resign at L5?",
        "What health insurance coverage is provided?",
        "How does the annual performance review work?",
        "What are the travel expense limits for L4?",
    ]
    col1, col2 = st.columns(2)
    for i, s in enumerate(suggestions):
        with (col1 if i % 2 == 0 else col2):
            if st.button(s, key=f"s{i}", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": s})
                with st.spinner("Searching HR policies..."):
                    result = ask(s)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result["sources"],
                    "is_hr": result["is_hr"],
                })
                st.rerun()

if prompt := st.chat_input("Ask about leave, salary, benefits, WFH, performance..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Searching HR policies..."):
            result = ask(prompt)
        st.markdown(result["answer"])
        if result["sources"]:
            st.caption("Sources: " + " | ".join(result["sources"]))
    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"],
        "is_hr": result["is_hr"],
    })
