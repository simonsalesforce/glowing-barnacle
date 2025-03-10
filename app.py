import os
import io
import asyncio
import time
import subprocess
import torch
import streamlit as st
from docx import Document
from faster_whisper import WhisperModel
from transformers import pipeline

# ----- 🛠 Force Install Missing Dependencies (If Needed) -----
try:
    import torch
except ImportError:
    st.write("🚀 Installing missing dependencies... This may take a moment.")
    subprocess.run(["pip", "install", "torch==2.2.0", "torchaudio==2.2.0", "torchvision==0.16.0"], check=True)
    import torch  # Retry import after installation

try:
    import faster_whisper
except ImportError:
    subprocess.run(["pip", "install", "faster-whisper==0.9.0"], check=True)
    import faster_whisper

try:
    import transformers
except ImportError:
    subprocess.run(["pip", "install", "transformers==4.38.0"], check=True)
    import transformers

# ----- 🛠 Fix Event Loop Issue -----
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# ----- 🖥 Debugging Info -----
st.write("🔍 Python Version:", os.popen("python --version").read().strip())
st.write("🔍 Torch Version:", torch.__version__)

# ----- 🌟 App Title -----
st.title("Education & Employers Audio Wizard")

# ----- 📌 Caching Models -----
@st.cache_resource(show_spinner=False)
def load_whisper_model(model_size="small"):
    return WhisperModel(model_size, device="cpu", compute_type="float32")  # Force float32

@st.cache_resource(show_spinner=False)
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn", device=-1)

# ----- 🔄 Initialize Session State for Outputs -----
if "transcript_text" not in st.session_state:
    st.session_state.transcript_text = None
if "transcript_bytes" not in st.session_state:
    st.session_state.transcript_bytes = None
if "summary_bytes" not in st.session_state:
    st.session_state.summary_bytes = None

# ----- 📂 Upload Audio File -----
uploaded_audio = st.file_uploader("Upload Audio File (MP3, WAV, M4A)", type=["mp3", "wav", "m4a"])
if uploaded_audio is not None:
    audio_ext = uploaded_audio.name.split('.')[-1]
    audio_path = f"temp_audio.{audio_ext}"
    with open(audio_path, "wb") as f:
        f.write(uploaded_audio.read())
    st.success("✅ Audio Uploaded! Click 'Transcribe Audio' to process.")

    # ----- 🎙 Transcribe Audio using faster-whisper -----
    if st.button("Transcribe Audio", key="transcribe"):
        with st.spinner("🔍 Transcribing..."):
            try:
                whisper_model = load_whisper_model("small")
                segments, info = whisper_model.transcribe(audio_path)
                transcript = " ".join(segment.text for segment in segments)
                st.session_state.transcript_text = transcript
                st.success("✅ Transcription Complete!")
            except Exception as e:
                st.error(f"❌ Error in transcription: {e}")

    # ----- ✂️ Options After Transcription -----
    if st.session_state.transcript_text:
        st.subheader("Transcript Options")
        if st.button("Summarize Transcript", key="summarize"):
            with st.spinner("📝 Summarizing..."):
                try:
                    summarizer = load_summarizer()
                    summary_output = summarizer(
                        st.session_state.transcript_text,
                        max_length=min(400, len(st.session_state.transcript_text) // 2),
                        min_length=150,
                        do_sample=False
                    )
                    summary_text = summary_output[0]['summary_text']
                    st.success("✅ Summary Generated!")
                    st.text_area("Summary", summary_text)
                except Exception as e:
                    st.error(f"❌ Error in summarization: {e}")

# ----- 🧹 Cleanup: Remove Temporary Audio File -----
if "audio_path" in locals() and os.path.exists(audio_path):
    os.remove(audio_path)

# ----- 🔻 Footer -----
st.markdown("<p style='text-align: center; font-size: 14px; color: gray;'>powered by Tea</p>", unsafe_allow_html=True)
