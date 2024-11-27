import streamlit as st
import sounddevice as sd
import numpy as np
import io
import soundfile as sf
from utils import TransformerModel, CNN2DFeatureExtractor, PositionalEncoding, SpeechRecognitionModel, sample_rate, preprocess, generate  # Your model functions
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
SpeechRecognitionModel(
    d_model=640, nhead=2, num_encoders=4, num_decoders=1, dim_feedforward=640,
    cnn_mode='simple', inplanes=32, planes=32,
    n_mels=80
    )

model = torch.load("model.pt", map_location=torch.device(device))
vocab = torch.load("vocab.pt", map_location=torch.device(device))
vocab.set_default_index(vocab['#'])

# Function to record audio
def record_audio(duration=5, sample_rate=sample_rate):
    st.info(f"Recording audio for {duration} seconds...")
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype="float32")
    sd.wait()  # Wait until recording is finished
    st.success("Recording completed!")
    return recording, sample_rate

# Convert the NumPy array to a WAV file in memory
def numpy_to_wav(audio_data, sample_rate):
    wav_io = io.BytesIO()
    sf.write(wav_io, audio_data, sample_rate, format='WAV')
    wav_io.seek(0)  # Reset pointer to the start of the BytesIO buffer
    return wav_io

st.set_page_config(
    page_title="Speech Recognition Web App",
    page_icon="🎙️",
)

# Initial app settings
st.title("Audio-to-Text Conversion")
st.markdown("""
Upload an audio file or record your voice to generate the text.
""")
st.write("---")

tab1, tab2 = st.tabs(['UPLOAD audio file', 'Record audio'])

# Tab for uploading an audio file
with tab1:
    uploaded_file = st.file_uploader("Upload your audio file:", type=["wav", "mp3", "ogg"])
    if uploaded_file:
        audio_data = uploaded_file.read() # Read the binary data
        st.audio(audio_data)  # Play the uploaded audio

        buffer = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32767.0
        audio_data = torch.from_numpy(buffer)
        # Preprocess and pass the data to the model
        with st.spinner("Processing uploaded audio..."):
            processed_audio = preprocess(audio_data)  # Custom preprocessing for your model
            generated_text = generate(model=model, vocab=vocab, audio=processed_audio)
        st.success("Processing complete!")
        st.write(f"**Generated Text:**\n\n{generated_text}")

# Tab for recording an audio file
with tab2:
    duration = st.slider("Recording duration (seconds):", 1, 10, 5)
    if st.button("Record Audio"):
        recorded_audio, fs = record_audio(duration=duration)
        # Convert NumPy audio to WAV format for playback
        wav_file = numpy_to_wav(recorded_audio, fs)
        st.audio(wav_file, format="audio/wav")  # Play the recorded audio
        # Pass the recorded audio to the model directly
        with st.spinner("Processing recorded audio..."):
            processed_audio = preprocess(recorded_audio)  # Custom preprocessing for your model
            generated_text = generate(model=model, vocab=vocab, audio=processed_audio)
        st.success("Processing complete!")
        st.write(f"**Generated Text:**\n\n{generated_text}")
