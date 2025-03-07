import os
import io
import torch
import streamlit as st
from docx import Document
import ollama
from faster_whisper import WhisperModel

# ----- Environment Fixes -----
# Force CPU-only operation and FP32
os.environ["TORCH_CPU_ONLY"] = "1"
torch.set_default_dtype(torch.float32)

# Ensure ffmpeg is available (adjust path if needed)
os.environ["PATH"] += os.pathsep + "/usr/bin/"

st.title("🎤 Education & Employers Audio Wizard")

# Initialize session state for transcript and summary
if "transcript_text" not in st.session_state:
    st.session_state.transcript_text = None
if "transcript_bytes" not in st.session_state:
    st.session_state.transcript_bytes = None
if "summary_bytes" not in st.session_state:
    st.session_state.summary_bytes = None

# ----- Upload Audio File -----
uploaded_audio = st.file_uploader("Upload Audio File (MP3, WAV, M4A)", type=["mp3", "wav", "m4a"])
if uploaded_audio is not None:
    # Save the uploaded file temporarily to disk
    audio_ext = uploaded_audio.name.split('.')[-1]
    audio_path = f"temp_audio.{audio_ext}"
    with open(audio_path, "wb") as f:
        f.write(uploaded_audio.read())
    st.success("✅ Audio Uploaded! Click 'Transcribe Audio' to process.")

    # ----- Transcribe Audio with faster-whisper -----
    if st.button("Transcribe Audio", key="transcribe"):
        with st.spinner("🔍 Transcribing..."):
            try:
                model = WhisperModel("small", device="cpu")
                segments, info = model.transcribe(audio_path)
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
                        prompt = f"""
Convert this transcript into a **highly detailed multi-page summary**.
- Expand all discussions fully.
- Use section headings and paragraphs.
- Make the summary at least **2,000 words**.

Transcript:
{st.session_state.transcript_text}
"""
                        # Use Mistral for summarization
                        response = ollama.chat(model="mistral", messages=[{"role": "user", "content": prompt}])
                        summary_text = response["message"]["content"]

                        # Generate a Word document for the summary in memory
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

        # Provide download button for summary if available
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
