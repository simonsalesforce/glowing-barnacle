import os
import io
import time
import torch
import streamlit as st
from docx import Document
from faster_whisper import WhisperModel
from transformers import pipeline

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
    # Use the TII Falcon-7B-Instruct model as an alternative instruct model.
    return pipeline(
        "text2text-generation",
        model="tiiuae/falcon-7b-instruct",
        device=-1  # Use CPU
    )

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

                        # Single-step prompt for a detailed report using Falcon instruct model
                        prompt = (
                            "You are an expert meeting summarizer. Using the transcript below, generate a comprehensive, detailed report "
                            "of about 1000 words. The report should include the following sections:\n\n"
                            "1. Meeting Overview\n"
                            "2. Detailed Discussion Points\n"
                            "3. Key Questions and Interactions\n"
                            "4. Decisions and Next Steps\n"
                            "5. Additional Insights and Recommendations\n\n"
                            "Please ensure that the report is well-structured, coherent, and provides actionable insights.\n\n"
                            "Transcript:\n" + st.session_state.transcript_text
                        )
                        summary_output = summarizer(
                            prompt,
                            max_length=1024,
                            min_length=512,
                            do_sample=True,
                            temperature=0.7
                        )
                        summary_text = summary_output[0]['generated_text']

                        # Simulate progress
                        progress_bar = st.progress(0)
                        for percent in range(0, 101, 10):
                            progress_bar.progress(percent)
                            time.sleep(0.1)

                        # Generate a Word document in memory for the summary
                        summary_doc = Document()
                        summary_doc.add_heading("Meeting Summary", level=1)
                        summary_doc.add_paragraph(summary_text)
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
st.markdown("<p style='text-align: center; font-size: 14px; color: gray;'>powered by Tea</p>", unsafe_allow_html=True)
