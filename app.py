import streamlit as st

from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.chat_engine import ContextChatEngine
from llama_index.core.memory import Memory
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq


# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="CarePulse AI",
    page_icon="🏥",
    layout="wide"
)


# ---------------- LIGHT HEALTHCARE THEME ----------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #F8FBFF 0%, #EEF7FF 50%, #F8FAFC 100%);
    color: #0F172A;
}

section[data-testid="stSidebar"] {
    background-color: #EAF6FF;
}

section[data-testid="stSidebar"] * {
    color: #0F172A !important;
}

.main-title {
    text-align: center;
    color: #1565C0;
    font-size: 44px;
    font-weight: 800;
    margin-bottom: 2px;
}

.sub-title {
    text-align: center;
    color: #0F172A;
    font-size: 22px;
    font-weight: 600;
}

.description {
    text-align: center;
    color: #475569;
    font-size: 16px;
    margin-bottom: 20px;
}

.question-title {
    color: #1565C0;
    font-weight: 700;
    font-size: 18px;
    margin-top: 10px;
}

.info-text {
    color: #475569;
    font-size: 14px;
    margin-bottom: 20px;
}

[data-testid="stChatInput"] {
    background-color: white !important;
    border-radius: 14px;
    border: 1px solid #BFDBFE;
}

[data-testid="stChatInput"] textarea {
    color: #000000 !important;
}

/* User and Assistant Messages */
[data-testid="stChatMessage"] {
    background-color: rgba(255, 255, 255, 0.95);
    border: 1px solid #DBEAFE;
    border-radius: 16px;
    padding: 10px;
    color: #000000 !important;
}

[data-testid="stChatMessage"] * {
    color: #000000 !important;
}

/* Markdown text inside chat */
.stMarkdown {
    color: #000000 !important;
}

.stButton > button {
    background-color: #1565C0;
    color: white;
    border-radius: 12px;
    border: none;
    padding: 8px 14px;
    font-weight: 600;
}

.stButton > button:hover {
    background-color: #0D47A1;
    color: white;
}

.warning-text {
    text-align: center;
    color: #92400E;
    background-color: #FEF3C7;
    padding: 10px;
    border-radius: 12px;
    font-size: 14px;
    margin-bottom: 20px;
}

.footer {
    text-align: center;
    color: #64748B;
    font-size: 13px;
}
</style>
""", unsafe_allow_html=True)


# ---------------- API KEY ----------------
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]


# ---------------- LOAD RESOURCES ----------------
@st.cache_resource
def load_resources():

    llm = Groq(
        model="llama-3.3-70b-versatile",
        api_key=GROQ_API_KEY,
        temperature=0.1,
        max_tokens=650
    )

    embeddings = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        cache_folder="./embedding_model/",
    )

    storage_context = StorageContext.from_defaults(
        persist_dir="./vector_index"
    )

    vector_index = load_index_from_storage(
        storage_context,
        embed_model=embeddings,
    )

    retriever = vector_index.as_retriever(similarity_top_k=5)

    prefix_messages = [
        ChatMessage(
            role=MessageRole.SYSTEM,
            content="""
You are CarePulse AI, an Emergency Department assistant.

Rules:
- Use ONLY the CarePulse knowledge base context.
- Do NOT hallucinate.
- Do NOT provide medical diagnosis, treatment, medicine, or emergency advice.
- Explain ER visit demand, patient wait time, triage, patient flow, weather, holidays, and time patterns.
- If the answer is not found in the knowledge base, say:
  "I do not have that information in the CarePulse knowledge base."
- Keep answers clear, structured, and easy to understand.
- Use bullet points when useful.
"""
        )
    ]

    return llm, retriever, prefix_messages


llm, retriever, prefix_messages = load_resources()


# ---------------- MEMORY ----------------
if "memory" not in st.session_state:
    st.session_state.memory = Memory.from_defaults(token_limit=2500)

if "messages" not in st.session_state:
    st.session_state.messages = []


# ---------------- CHAT ENGINE ----------------
rag_bot = ContextChatEngine(
    llm=llm,
    retriever=retriever,
    memory=st.session_state.memory,
    prefix_messages=prefix_messages,
)


# ---------------- HEADER ----------------
# ---------------- HEADER ----------------
col1, col2, col3 = st.columns([2, 1, 2])

with col2:
    st.markdown(
        """
        <div style="padding-left:90px;">
        """,
        unsafe_allow_html=True
    )

    st.image("image/logo.jpg", width=250)

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='main-title'>CarePulse AI</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>CarePulse Health Assistant</div>", unsafe_allow_html=True)
# Heartbeat Line Image
col1, col2, col3 = st.columns([2,1,2])

with col2:
    st.image("image/logo1.jpg",width=320)
    
st.markdown(
    "<div class='description'>Your Smart Companion for Emergency Department Insights</div>",
    unsafe_allow_html=True
)

st.markdown(
    """
    <div style="
        background:#FEF2F2;
        border-left:6px solid #DC2626;
        border-radius:10px;
        padding:12px;
        color:#7F1D1D;
        font-size:14px;
        text-align:center;
        margin-bottom:20px;
    ">
    🚨 <b>Educational Use Only</b><br>
    No medical diagnosis, treatment, medication, or emergency advice is provided.
    </div>
    """,
    unsafe_allow_html=True
)

st.divider()


# ---------------- SUGGESTED QUESTIONS ----------------
st.markdown(
    """
    <div class='question-title' style='margin-bottom:25px;'>
        Explore Emergency Department Insights
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class='info-text'>
        Select a topic below or ask your own question.
    </div>
    """,
    unsafe_allow_html=True
)

suggested_prompt = None

q1, q2, q3, q4 = st.columns(4)

with q1:
    if st.button("🚑 What is triage?", use_container_width=True):
        suggested_prompt = "What is triage?"

with q2:
    if st.button("🌙 Why is ER busy at night?", use_container_width=True):
        suggested_prompt = "Why is ER busy at night?"

with q3:
    if st.button("⏳ What affects wait time?", use_container_width=True):
        suggested_prompt = "What factors affect patient wait time?"

with q4:
    if st.button("🌦️ Weather impact", use_container_width=True):
        suggested_prompt = "How does weather affect ER visits?"


st.write("")

q5, q6, q7, q8 = st.columns(4)

with q5:
    if st.button("📅 Holiday impact", use_container_width=True):
        suggested_prompt = "How do holidays affect ER demand?"

with q6:
    if st.button("📈 What is ER demand?", use_container_width=True):
        suggested_prompt = "What is ER demand?"

with q7:
    if st.button("🕒 Patient wait time", use_container_width=True):
        suggested_prompt = "What is patient wait time?"

with q8:
    if st.button("🤖 ER demand prediction", use_container_width=True):
        suggested_prompt = "Can ER demand be predicted?"

# ---------------- DISPLAY CHAT HISTORY ----------------
for msg in st.session_state.messages:
    avatar = "👤" if msg["role"] == "user" else "image/logo.jpg"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])


# ---------------- CHAT INPUT ----------------
user_prompt = st.chat_input(
    "Ask about ER visits, triage, hospital workflow, or patient wait time..."
)

final_prompt = suggested_prompt or user_prompt

if final_prompt:
    st.session_state.messages.append(
        {"role": "user", "content": final_prompt}
    )

    with st.chat_message("user", avatar="👤"):
        st.markdown(final_prompt)

    with st.chat_message("assistant", avatar="image/logo.jpg"):
        with st.spinner("Searching knowledge base..."):
            response = rag_bot.chat(final_prompt)
            answer = response.response
            st.markdown(answer)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )


# ---------------- FOOTER ----------------
st.divider()

st.markdown(
    "<div class='footer'>CarePulse AI | RAG-based Emergency Department Assistant</div>",
    unsafe_allow_html=True
)

st.write("")

col1, col2 = st.columns([8, 1])

with col2:
    if st.button("🗑 Reset"):
        st.session_state.memory = Memory.from_defaults(token_limit=2500)
        st.session_state.messages = []
        st.rerun() 
