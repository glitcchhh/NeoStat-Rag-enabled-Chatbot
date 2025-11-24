# app.py
import streamlit as st
import os
from streamlit_mic_recorder import mic_recorder
import base64
import requests

from utils.embedding_utils import chunk_text, build_vector_store, retrieve
from utils.web_search import web_search
from utils.helpers import safe_call
from models.llm import generate_response
from config.config import VECTOR_STORE_PATH


# Perplexity Voice-to-Text
PPLX_API_KEY = os.getenv("PPLX_API_KEY")
PERPLEXITY_STT_URL = "https://api.perplexity.ai/audio/transcriptions"

def perplexity_speech_to_text(audio_bytes):
    """
    Convert recorded audio bytes into text using Perplexity Sonar-Pro STT.
    """
    files = {
        "file": ("audio.wav", audio_bytes, "audio/wav")
    }
    data = {
        "model": "sonar-pro",
    }
    headers = {
        "Authorization": f"Bearer {PPLX_API_KEY}"
    }

    response = requests.post(PERPLEXITY_STT_URL, files=files, data=data, headers=headers)
    response.raise_for_status()

    return response.json().get("text", "")


# STREAMLIT PAGE CONFIG
st.set_page_config(
    page_title="NeoStats Chatbot — RAG + Voice + Web Search",
    layout="wide"
)

st.title("🎤 NeoStats — RAG-enabled Chatbot with Voice Input")


# SIDEBAR: UPLOAD DOCUMENTS
st.sidebar.title("Settings")
st.sidebar.markdown("Upload documents, enable RAG, Web Search, and Voice-to-Text.")

st.sidebar.header("1) Upload Documents")

uploaded_files = st.sidebar.file_uploader(
    "Upload .txt / .md / text-based PDF files",
    type=["txt", "md", "pdf"],
    accept_multiple_files=True
)

if st.sidebar.button("Build / Refresh Vector Store"):
    all_docs, metas = [], []

    for f in uploaded_files:
        try:
            raw = f.read()
            try:
                text = raw.decode("utf-8")
            except:
                text = str(raw)

            chunks = chunk_text(text)
            for c in chunks:
                all_docs.append(c)
                metas.append({"filename": f.name})

        except Exception as e:
            st.sidebar.error(f"Error reading {f.name}: {e}")

    if all_docs:
        build_vector_store(all_docs, metas, VECTOR_STORE_PATH)
        st.sidebar.success(f"Vector store built with {len(all_docs)} chunks!")
    else:
        st.sidebar.warning("Upload files first!")


# Sidebar Options
st.sidebar.header("Options")

use_web_search = st.sidebar.checkbox("Enable Live Web Search", value=True)

mode = st.sidebar.radio(
    "Response Mode",
    ["concise", "detailed"],
    index=1
)


# Conversation State
if "history" not in st.session_state:
    st.session_state.history = []

if "voice_text" not in st.session_state:
    st.session_state.voice_text = ""


# VOICE INPUT
st.header("🎤 Voice Input (Speak your question)")

audio_recording = mic_recorder(
    start_prompt="🎙️ Start Recording",
    stop_prompt="⏹️ Stop Recording",
    key="voice_recorder"
)

voice_question = ""

if audio_recording is not None:
    audio_bytes = audio_recording["audio"]
    st.audio(audio_bytes, format="audio/wav")

    st.write("⏳ Converting your speech to text using Perplexity Sonar-Pro...")

    try:
        voice_question = perplexity_speech_to_text(audio_bytes)
        st.session_state.voice_text = voice_question
        st.success(f"**You said:** {voice_question}")

    except Exception as e:
        st.error(f"Voice-to-text failed: {e}")


# TEXT INPUT
st.header("💬 Ask a Question")

user_input = st.text_area(
    "Your question:",
    height=120,
    value=st.session_state.voice_text
)


# PROCESS QUERY
if st.button("Send") and user_input.strip():

    # RAG Retrieval
    retrieved_docs = safe_call(retrieve, user_input, 4, VECTOR_STORE_PATH) or []
    context_text = "\n\n".join(
        [f"(score={r['score']:.3f})\n{r['doc']}" for r in retrieved_docs]
    )

    # Web Search
    web_data = ""
    if use_web_search:
        search_output = safe_call(web_search, user_input)
        if search_output:
            if isinstance(search_output, list):
                search_output = "\n\n".join([str(item) for item in search_output])
            web_data = "\n\n[WEB SEARCH RESULTS]\n" + str(search_output)

    # LLM prompt
    final_prompt = f"""
You are an AI assistant with RAG + Web Search.

User Question:
{user_input}

Retrieved Document Context:
{context_text}

Web Search Context:
{web_data}

Response Mode: {mode}

Provide the best possible answer.
"""

    # Generate answer
    answer = safe_call(generate_response, final_prompt, mode)

    st.session_state.history.append({"user": user_input, "bot": answer})

    st.subheader("Response:")
    st.write(answer)



# SIDEBAR CHAT HISTORY
st.sidebar.header("Chat History")
for msg in reversed(st.session_state.history):
    st.sidebar.markdown(f"**You:** {msg['user']}")
    st.sidebar.markdown(f"**Bot:** {msg['bot']}")
    st.sidebar.markdown("---")
