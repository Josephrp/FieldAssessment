import streamlit as st
import numpy as np
from audiorecorder import audiorecorder
import whisper
import os
import streamlit.components.v1 as components
from gradio_client import Client
import tempfile
import io

# Set the token as an environment variable
os.environ["YOUR_API_TOKEN"] = "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"

# Retrieve the token
token = os.environ["YOUR_API_TOKEN"]

# Define the Gradio server URL
gradio_server_url = "https://teamtonic-llavaapi.hf.space/--replicas/5sg7p/"


# Rest of your code...

st.title("Team Tonic Demo")
st.subheader("Take a picture first and speak your request for the model")

image = st.camera_input("Camera input")

audio = audiorecorder("Click to record audio", "Click to stop recording")

if len(audio) > 0:
    audio_data = audio.export().read()  # Read audio data as bytes
    audio_bytes_io = io.BytesIO(audio_data)  # Create a BytesIO object

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as audio_file:
        audio_file_path = audio_file.name
        audio_file.write(audio_bytes_io.read())  # Write audio data to the temporary file

    st.audio(audio_file_path, format="audio/wav")  # Show audio only if it's recorded

    submit_button = st.button("Use this audio")

    if submit_button:
        # Load the Whisper model
        model = whisper.load_model("base")

        result = model.transcribe(audio_file_path)
        st.info("Transcribing...")

        st.success("Transcription complete")
        transcript = result['text']

        with st.expander("See transcript"):
            st.markdown(transcript)

        st.text("Sending image and request to the model. Please wait...")
        # Define the LLavA Client
        # Define the LLavA Client with the Gradio server URL
        client = Client(src=gradio_server_url)

        # Sample API call to LLavA
        result_llava = client.predict(
            transcript,
            image,
            "Crop",
            fn_index=7
        )


        # Display LLavA result in a text box
        st.subheader("LLavA Result:")
        st.json(result_llava)

        # Text-to-Speech Translation
        tts_client = Client("https://facebook-seamless-m4t.hf.space/")
        tts_result = tts_client.predict(
            "T2ST (Text to Speech translation)",
            result_llava,  # Replacing "howdy" with LLavA result
            "english",
            "english"
        )

        # Display TTS result in a text box
        st.subheader("Text-to-Speech Translation Result:")
        st.json(tts_result)

        # Autoplay TTS audio
        audio_url = tts_result["audio_url"]
        st.audio(audio_url, format="audio/wav", start_time=0, key="tts_audio")

    # Add a reset button to clear the interface
    if st.button("Reset"):
        st.experimental_rerun()
