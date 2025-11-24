import streamlit as st
import os

from utils.embedding_utils import chunk_text, build_vector_store, retrieve
from utils.web_search import web_search
from utils.helpers import safe_call
from models.llm import generate_response
from config.config import VECTOR_STORE_PATH


# Streamlit Page Config
st.set_page_config(
    page_title="NeoStats Chatbot — RAG + Live Web Search",
    layout="wide"
)

st.title("❤️ NeoStats — RAG-enabled Chatbot")


# Sidebar — Upload, Settings
st.sidebar.title("Settings")
st.sidebar.markdown("Upload documents, enable RAG, web search, and set response mode.")


# Document Upload Section 
st.sidebar.header("1) Upload Documents")

uploaded_files = st.sidebar.file_uploader(
    "Upload .txt / .md / text-based PDF files",
    type=["txt", "md", "pdf"],
    accept_multiple_files=True
)

if st.sidebar.button("Build / Refresh Vector Store"):
    all_docs = []
    metas = []

    for f in uploaded_files:
        try:
            raw = f.read()
            try:
                text = raw.decode("utf-8")
            except Exception:
                # fallback for binary / pdf text extraction (text-only pdfs)
                text = str(raw)

            chunks = chunk_text(text)
            for c in chunks:
                all_docs.append(c)
                metas.append({"filename": f.name})

        except Exception as e:
            st.sidebar.error(f"Failed to read {f.name}: {e}")

    if all_docs:
        safe_call(build_vector_store, all_docs, metas, VECTOR_STORE_PATH)
        st.sidebar.success(f"Vector store built with {len(all_docs)} chunks")
    else:
        st.sidebar.warning("Upload files before building the vector store.")


# Options Section
st.sidebar.header("Options")

use_web_search = st.sidebar.checkbox("Enable Live Web Search", value=True)

mode = st.sidebar.radio(
    "Response Mode",
    options=["concise", "detailed"],
    index=1
)



# Conversation State
if "history" not in st.session_state:
    st.session_state.history = []


# Chat Input Area
st.header("Ask a Question")

col1, col2 = st.columns([3, 1])

with col1:
    user_input = st.text_area("Your question:", height=120)

    if st.button("Send") and user_input.strip():

        #  RAG Retrieval logic 
        retrieved_docs = safe_call(retrieve, user_input, 4, VECTOR_STORE_PATH) or []
        context_text = "\n\n".join(
            [f"(score={r['score']:.3f})\n{r['doc']}" for r in retrieved_docs]
        )

        # Web Search Logic 
        web_data = ""
        if use_web_search:

            # If no retrieved RAG chunks, rely fully on search
            if len(retrieved_docs) == 0:
                search_output = safe_call(web_search, user_input)
                if search_output:
                    web_data = "\n\n[WEB SEARCH RESULTS]\n" + search_output

            # If RAG exists, optionally add supplementary context
            else:
                search_output = safe_call(web_search, user_input)
                if search_output:
                    web_data = "\n\n[SUPPLEMENTARY SEARCH]\n" + search_output

        # LLM Final Prompt
        final_prompt = f"""
You are an AI assistant with RAG + Web Search capability.

User Question:
{user_input}

Retrieved Document Context:
{context_text}

Web Search Context:
{web_data}

Response Mode: {mode}

Provide the best possible answer.
"""

        # Generate Answer
        answer = safe_call(generate_response, final_prompt, mode)

        # Save in session
        st.session_state.history.append({"user": user_input, "bot": answer})

        # Display response
        st.subheader("Response:")
        st.write(answer)


# Sidebar: Conversation History
with col2:
    st.subheader("Chat History")
    for msg in reversed(st.session_state.history):
        st.markdown("**You:** " + msg["user"])
        st.markdown("**Bot:** " + msg["bot"])
        st.markdown("---")
