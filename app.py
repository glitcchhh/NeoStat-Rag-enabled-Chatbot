# app.py
import streamlit as st
import os
from streamlit_mic_recorder import mic_recorder
import base64
import requests

from google.cloud import speech  # GOOGLE STT

from utils.embedding_utils import chunk_text, build_vector_store, retrieve
from utils.web_search import web_search
from utils.helpers import safe_call
from models.llm import generate_response
from config.config import VECTOR_STORE_PATH



# ---------------- GOOGLE SPEECH-TO-TEXT ---------------- #

def google_speech_to_text(audio_bytes):
    """
    Convert recorded audio bytes into text using Google Speech-to-Text.
    """
    client = speech.SpeechClient()

    audio = speech.RecognitionAudio(content=audio_bytes)

    config = speech.RecognitionConfig(
        language_code="en-US",
        enable_automatic_punctuation=True,
        audio_channel_count=1
    )

    response = client.recognize(config=config, audio=audio)

    if len(response.results) == 0:
        return ""

    transcript = response.results[0].alternatives[0].transcript
    return transcript



# ---------------- STREAMLIT CONFIG ---------------- #

st.set_page_config(
    page_title="NeoStats Chatbot — RAG + Voice Search + Web Search",
    layout="wide"
)

st.title("NeoStats — RAG-enabled Chatbot with Voice Input")


# ---------------- SIDEBAR: DOCUMENT UPLOAD ---------------- #

st.sidebar.title("Settings")
st.sidebar.markdown("Upload documents, enable RAG, Web Search, and Voice-to-Text.")

st.sidebar.header("1) Upload Documents")

uploaded_files = st.sidebar.file_uploader(
    "Upload .txt / .md / PDF files",
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


# Sidebar options
st.sidebar.header("Options")

use_web_search = st.sidebar.checkbox("Enable Live Web Search", value=True)

mode = st.sidebar.radio(
    "Response Mode",
    ["concise", "detailed"],
    index=1
)


# ---------------- SESSION STATE ---------------- #

if "history" not in st.session_state:
    st.session_state.history = []

if "voice_text" not in st.session_state:
    st.session_state.voice_text = ""


# ---------------- VOICE INPUT ---------------- #

st.header("🎤 Voice Input (Speak your question)")

audio_dict = mic_recorder(
    start_prompt="🎤 Start Recording",
    stop_prompt="⏹ Stop Recording"
)

voice_question = ""

if audio_dict and "bytes" in audio_dict:
    audio_bytes = audio_dict["bytes"]

    st.audio(audio_bytes, format="audio/wav")
    st.write("⏳ Converting speech to text using Google Speech-to-Text...")

    try:
        voice_question = google_speech_to_text(audio_bytes)
        st.session_state.voice_text = voice_question
        st.success(f"**You said:** {voice_question}")
    except Exception as e:
        st.error(f"Google STT failed: {e}")


# ---------------- TEXT INPUT ---------------- #

st.header("💬 Ask a Question")

user_input = st.text_area(
    "Your question:",
    height=120,
    value=st.session_state.voice_text
)


# ---------------- PROCESSING QUERY ---------------- #

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

    # LLM Prompt for Perplexity
    final_prompt = f"""
You are an AI assistant with RAG + Web Search.

User Question:
{user_input}

Retrieved Document Context:
{context_text}

Web Search Context:
{web_data}

Response Mode: {mode}

Give the best possible answer.
"""

    # Generate response with Perplexity
    answer = safe_call(generate_response, final_prompt, mode)

    st.session_state.history.append({"user": user_input, "bot": answer})

    st.subheader("Response:")
    st.write(answer)


# ---------------- SIDEBAR HISTORY ---------------- #

st.sidebar.header("Chat History")
for msg in reversed(st.session_state.history):
    st.sidebar.markdown(f"**You:** {msg['user']}")
    st.sidebar.markdown(f"**Bot:** {msg['bot']}")
    st.sidebar.markdown("---")
