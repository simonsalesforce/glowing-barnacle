import streamlit as st
import ollama
import whisper
import os
from docx import Document

os.environ["PATH"] += os.pathsep + "/usr/bin/"


st.title("🎤 Education and Employers Audio Transcriber")

# 🔹 Upload Audio File
uploaded_audio = st.file_uploader("Upload Audio File (MP3, WAV, M4A)", type=["mp3", "wav", "m4a"])
if uploaded_audio is not None:
    audio_ext = uploaded_audio.name.split('.')[-1]
    audio_path = f"temp_audio.{audio_ext}"

    # Save uploaded file
    with open(audio_path, "wb") as f:
        f.write(uploaded_audio.read())
    st.success("✅ Audio Uploaded! Click 'Transcribe' to process.")

    # 🔹 Transcribe Audio with Whisper
    if st.button("Transcribe Audio"):
        with st.spinner("🔍 Transcribing..."):
            try:
                model = whisper.load_model("small")  # Use Whisper-Small (change to "medium" or "large" if needed)
                result = model.transcribe(audio_path)
                transcript_text = result["text"]

                # Save transcript as Word file
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

    # 🔹 Summarize Transcript
    if "transcript_text" in locals() and st.button("Summarize Transcript"):
        with st.spinner("📝 Summarizing..."):
            try:
                prompt = f"""
                Convert this transcript into a **highly detailed multi-page summary**.
                - Expand all discussions fully.
                - Use section headings and paragraphs.
                - Make the summary at least **2,000 words**.

                Transcript:
                {transcript_text}
                """

                response = ollama.chat(model="mistral", messages=[{"role": "user", "content": prompt}])
                summary_text = response["message"]["content"]

                # Save summary as Word file
                summary_doc = Document()
                summary_doc.add_heading("Meeting Summary", level=1)
                summary_doc.add_paragraph(summary_text)
                summary_path = "summary.docx"
                summary_doc.save(summary_path)

                st.success("✅ Summary Generated!")
                with open(summary_path, "rb") as f:
                    st.download_button("📥 Download Summary", f, "summary.docx")
            except Exception as e:
                st.error(f"❌ Error in summarization: {e}")

    # Clean up files after processing
    if os.path.exists(audio_path):
        os.remove(audio_path)
    if os.path.exists("audio_transcript.docx"):
        os.remove("audio_transcript.docx")
    if os.path.exists("summary.docx"):
        os.remove("summary.docx")
