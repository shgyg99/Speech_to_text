import streamlit as st
from pydub import AudioSegment
from pydub.playback import play
import os
import sounddevice as sd
from scipy.io.wavfile import write

# Function to process the audio file and generate text
def generate_text(audio_file_path):
    # Add your speech-to-text function here
    # For example, use a speech recognition model
    return f"Generated text for the file: {os.path.basename(audio_file_path)}"

# Function to record audio
def record_audio(duration=5, filename="recorded_audio.wav"):
    st.info(f"Recording audio for {duration} seconds...")
    fs = 44100  # Sampling rate
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=2, dtype="int16")
    sd.wait()  # Wait until recording is finished
    write(filename, fs, recording)  # Save the file as a WAV file
    st.success("Recording completed and saved!")

# Initial app settings
st.title("🎙️ Audio-to-Text Conversion")
st.write("Upload an audio file or record your voice to generate the text.")

# Upload audio file
uploaded_file = st.file_uploader("Upload your audio file:", type=["wav", "mp3", "ogg"])

if uploaded_file:
    # Save the uploaded audio file
    audio_file_path = f"temp_audio.{uploaded_file.name.split('.')[-1]}"
    with open(audio_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.audio(audio_file_path, format="audio/wav")
    
    # Process the audio and generate text
    if st.button("Generate Text"):
        with st.spinner("Processing audio..."):
            generated_text = generate_text(audio_file_path)
        st.success("Processing complete!")
        st.write(f"**Generated Text:**\n\n{generated_text}")

# Record audio UI
st.title("🎙️ Record Audio")
st.write("Press the button below to record your voice:")

if st.button("Record Audio"):
    record_audio()
    st.audio("recorded_audio.wav")
