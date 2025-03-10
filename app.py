import os
import torch
import asyncio
import streamlit as st
from docx import Document
from transformers import pipeline
from faster_whisper import WhisperModel

# ----- Environment Fixes -----
# Force CPU-only execution for both Torch and Whisper
os.environ["TORCH_CPU_ONLY"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU inference
torch.set_default_dtype(torch.float32)

# Ensure Streamlit starts correctly with an event loop
if not hasattr(asyncio, "get_running_loop"):
    asyncio.get_event_loop().close()
    asyncio.set_event_loop(asyncio.new_event_loop())

st.title("🎤 AI Audio Summarizer")

# ----- Session State Initialization -----
if "transcript_text" not in st.session_state:
    st.session_state.transcript_text = None

# ----- Upload Audio File -----
uploaded_audio = st.file_uploader("Upload Audio File (MP3, WAV, M4A)", type=["mp3", "wav", "m4a"])
if uploaded_audio is not None:
    audio_ext = uploaded_audio.name.split('.')[-1]
    audio_path = f"temp_audio.{audio_ext}"

    # Save the uploaded file to disk
    with open(audio_path, "wb") as f:
        f.write(uploaded_audio.read())
    st.success("✅ Audio Uploaded! Click 'Transcribe Audio' to process.")

    # ----- Transcribe Audio with faster-whisper -----
    if st.button("Transcribe Audio"):
        with st.spinner("🔍 Transcribing..."):
            try:
                # Initialize the faster-whisper model on CPU
                model = WhisperModel("small", device="cpu", compute_type="int8")
                segments, info = model.transcribe(audio_path)
                transcript_text = " ".join(segment.text for segment in segments)
                st.session_state.transcript_text = transcript_text  # store transcript in session state

                # Save the transcript to a Word document
                transcript_doc = Document()
                transcript_doc.add_heading("Audio Transcript", level=1)
                transcript_doc.add_paragraph(transcript_text)
                transcript_path = "audio_transcript.docx"
                transcript_doc.save(transcript_path)

                st.success("✅ Transcription Complete!")
                with open(transcript_path, "rb") as f:
                    st.download_button("📥 Download Transcript", f, "audio_transcript.docx")
            except Exception as e:
                st.error(f"❌ Error in transcription: {e}")

    # ----- Function to Chunk Transcript Text -----
    def chunk_text(text, max_chunk_size=1000):
        """
        Splits the text into chunks of at most max_chunk_size characters,
        trying to split at the nearest period.
        """
        chunks = []
        while len(text) > max_chunk_size:
            chunk = text[:max_chunk_size]
            last_period = chunk.rfind('.')
            if last_period == -1:
                last_period = max_chunk_size
            chunk = text[:last_period+1]
            chunks.append(chunk.strip())
            text = text[last_period+1:]
        if text.strip():
            chunks.append(text.strip())
        return chunks

    # ----- Summarize Transcript using Transformers -----
    if st.session_state.transcript_text and st.button("Summarize Transcript"):
        with st.spinner("📝 Summarizing..."):
            try:
                # Initialize the summarization pipeline
                summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

                # Chunk the transcript text to respect model input size limits
                chunks = chunk_text(st.session_state.transcript_text, max_chunk_size=1000)
                summaries = []
                for chunk in chunks:
                    summary = summarizer(chunk, max_length=300, min_length=100, do_sample=False)
                    summaries.append(summary[0]['summary_text'])

                # Combine summaries into a final summary
                final_summary = "\n\n".join(summaries)

                # Save the final summary to a Word document
                summary_doc = Document()
                summary_doc.add_heading("Meeting Summary", level=1)
                summary_doc.add_paragraph(final_summary)
                summary_path = "summary.docx"
                summary_doc.save(summary_path)

                st.success("✅ Summary Generated!")
                with open(summary_path, "rb") as f:
                    st.download_button("📥 Download Summary", f, "summary.docx")
            except Exception as e:
                st.error(f"❌ Error in summarization: {e}")

    # ----- Cleanup Temporary Files -----
    for filename in [audio_path, "audio_transcript.docx", "summary.docx"]:
        if os.path.exists(filename):
            os.remove(filename)
