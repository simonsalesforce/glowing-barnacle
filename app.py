import os
import io
import time
import torch
import streamlit as st
from docx import Document
from faster_whisper import WhisperModel
from transformers import pipeline, AutoTokenizer

# ----- Environment Fixes -----
os.environ["TORCH_CPU_ONLY"] = "1"
torch.set_default_dtype(torch.float32)
os.environ["PATH"] += os.pathsep + "/usr/bin/"

st.title("Education & Employers Audio Wizard")

# ----- Caching Models -----
@st.cache_resource(show_spinner=False)
def load_whisper_model(model_size="small"):
    return WhisperModel(model_size, device="cpu")

@st.cache_resource(show_spinner=False)
def load_summarizer():
    # Use a smaller summarization model for lower resource usage
    return pipeline(
        "text2text-generation",
        model="google/flan-t5-small",
        device=-1  # Use CPU
    )

# ----- Helper Function to Chunk Text -----
def chunk_text(text, max_words=300):
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words):
        chunks.append(" ".join(words[i:i+max_words]))
    return chunks

# ----- Initialize Session State for Outputs -----
if "transcript_text" not in st.session_state:
    st.session_state.transcript_text = None
if "transcript_bytes" not in st.session_state:
    st.session_state.transcript_bytes = None
if "summary_bytes" not in st.session_state:
    st.session_state.summary_bytes = None

# ----- Upload Audio File -----
uploaded_audio = st.file_uploader("Upload Audio File (MP3, WAV, M4A)", type=["mp3", "wav", "m4a"])
if uploaded_audio is not None:
    audio_ext = uploaded_audio.name.split('.')[-1]
    audio_path = f"temp_audio.{audio_ext}"
    with open(audio_path, "wb") as f:
        f.write(uploaded_audio.read())
    st.success("✅ Audio Uploaded! Click 'Transcribe Audio' to process.")

    # ----- Transcribe Audio using faster-whisper -----
    if st.button("Transcribe Audio", key="transcribe"):
        with st.spinner("🔍 Transcribing..."):
            try:
                progress_bar = st.progress(0)
                for percent in range(0, 51, 10):
                    progress_bar.progress(percent)
                    time.sleep(0.1)  # simulated progress

                whisper_model = load_whisper_model("small")
                segments, info = whisper_model.transcribe(audio_path)
                transcript = " ".join(segment.text for segment in segments)
                st.session_state.transcript_text = transcript

                # Generate a Word document in memory for the transcript
                transcript_doc = Document()
                transcript_doc.add_heading("Audio Transcript", level=1)
                transcript_doc.add_paragraph(transcript)
                transcript_buffer = io.BytesIO()
                transcript_doc.save(transcript_buffer)
                transcript_buffer.seek(0)
                st.session_state.transcript_bytes = transcript_buffer.getvalue()

                progress_bar.progress(100)
                st.success("✅ Transcription Complete!")
            except Exception as e:
                st.error(f"❌ Error in transcription: {e}")

    # ----- Options After Transcription -----
    if st.session_state.transcript_text:
        st.subheader("Transcript Options")
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="📥 Download Transcript",
                data=st.session_state.transcript_bytes,
                file_name="audio_transcript.docx",
                key="download_transcript"
            )
        with col2:
            if st.button("Summarize Transcript", key="summarize"):
                with st.spinner("📝 Summarizing..."):
                    try:
                        summarizer = load_summarizer()
                        # Chunk the transcript to avoid resource limits
                        chunks = chunk_text(st.session_state.transcript_text, max_words=300)
                        chunk_summaries = []
                        sum_progress = st.progress(0)
                        total_chunks = len(chunks)
                        for idx, chunk in enumerate(chunks):
                            prompt = f"""
Summarize the following transcript excerpt in a concise manner:
{chunk}
"""
                            summary_chunk = summarizer(prompt, max_length=256, min_length=128, do_sample=False)
                            chunk_summaries.append(summary_chunk[0]['generated_text'])
                            sum_progress.progress(int((idx+1)/total_chunks * 50))
                            time.sleep(0.1)
                        
                        # Combine individual summaries and further summarize if needed
                        combined_summary = "\n".join(chunk_summaries)
                        final_prompt = f"""
Combine and refine the following summaries into a detailed, multi-page report (around 1000 words).
Include:
1. Meeting Overview
2. Detailed Discussion Points
3. Key Questions and Interactions
4. Decisions and Next Steps

Summaries:
{combined_summary}
"""
                        final_summary_output = summarizer(final_prompt, max_length=1024, min_length=512, do_sample=False)
                        final_summary_text = final_summary_output[0]['generated_text']

                        for percent in range(51, 101, 10):
                            sum_progress.progress(percent)
                            time.sleep(0.1)

                        # Generate a Word document in memory for the summary
                        summary_doc = Document()
                        summary_doc.add_heading("Meeting Summary", level=1)
                        summary_doc.add_paragraph(final_summary_text)
                        summary_buffer = io.BytesIO()
                        summary_doc.save(summary_buffer)
                        summary_buffer.seek(0)
                        st.session_state.summary_bytes = summary_buffer.getvalue()

                        st.success("✅ Summary Generated!")
                    except Exception as e:
                        st.error(f"❌ Error in summarization: {e}")

        if st.session_state.summary_bytes:
            st.download_button(
                label="📥 Download Summary",
                data=st.session_state.summary_bytes,
                file_name="summary.docx",
                key="download_summary"
            )

    # ----- Cleanup: Remove temporary audio file -----
    if os.path.exists(audio_path):
        os.remove(audio_path)

# ----- Footer Strapline -----
st.markdown("<p style='text-align: center; font-size: 14px; color: gray;'>Built by Simon, powered by Tea</p>", unsafe_allow_html=True)
