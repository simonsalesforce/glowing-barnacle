import os
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

st.title("🎤 AI Audio Summarizer")

# Initialize session state for transcript if not present
if "transcript_text" not in st.session_state:
    st.session_state.transcript_text = None

# ----- Upload Audio File -----
uploaded_audio = st.file_uploader("Upload Audio File (MP3, WAV, M4A)", type=["mp3", "wav", "m4a"])
if uploaded_audio is not None:
    audio_ext = uploaded_audio.name.split('.')[-1]
    audio_path = f"temp_audio.{audio_ext}"

    # Save the uploaded file
    with open(audio_path, "wb") as f:
        f.write(uploaded_audio.read())
    st.success("✅ Audio Uploaded! Click 'Transcribe Audio' to process.")

    # ----- Transcribe Audio with faster-whisper -----
    if st.button("Transcribe Audio"):
        with st.spinner("🔍 Transcribing..."):
            try:
                # Initialize faster-whisper on CPU (choose "small", "medium", etc.)
                model = WhisperModel("small", device="cpu")
                segments, info = model.transcribe(audio_path)
                transcript = " ".join(segment.text for segment in segments)
                st.session_state.transcript_text = transcript

                # Save transcript as a Word document
                transcript_doc = Document()
                transcript_doc.add_heading("Audio Transcript", level=1)
                transcript_doc.add_paragraph(transcript)
                transcript_path = "audio_transcript.docx"
                transcript_doc.save(transcript_path)

                st.success("✅ Transcription Complete!")
                st.download_button("📥 Download Transcript", open(transcript_path, "rb"), "audio_transcript.docx")
            except Exception as e:
                st.error(f"❌ Error in transcription: {e}")

    # ----- Options After Transcription -----
    if st.session_state.transcript_text:
        st.subheader("Transcript Options")
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="📥 Download Transcript",
                data=open("audio_transcript.docx", "rb").read(),
                file_name="audio_transcript.docx"
            )
        with col2:
            if st.button("Summarize Transcript"):
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
                        response = ollama.chat(model="mistral", messages=[{"role": "user", "content": prompt}])
                        summary_text = response["message"]["content"]

                        # Save summary as a Word document
                        summary_doc = Document()
                        summary_doc.add_heading("Meeting Summary", level=1)
                        summary_doc.add_paragraph(summary_text)
                        summary_path = "summary.docx"
                        summary_doc.save(summary_path)

                        st.success("✅ Summary Generated!")
                        st.download_button("📥 Download Summary", open(summary_path, "rb"), "summary.docx")
                    except Exception as e:
                        st.error(f"❌ Error in summarization: {e}")

    # ----- Cleanup Temporary Files -----
    for filename in [audio_path, "audio_transcript.docx", "summary.docx"]:
        if os.path.exists(filename):
            os.remove(filename)
