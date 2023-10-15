import streamlit as st
import numpy as np
from audiorecorder import audiorecorder
from gradio_client import Client
import whisper
import os
import streamlit.components.v1 as components

# Set the token as an environment variable
os.environ["YOUR_API_TOKEN"] = "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"

# Retrieve the token
token = os.environ["YOUR_API_TOKEN"]

st.title("Team Tonic Demo")
st.subheader("Take a picture first and speak your request for the model")

image = st.camera_input("Camera input")

audio = audiorecorder("Click to record audio", "Click to stop recording")

model = whisper.load_model("base")

client = Client("https://teamtonic-llavaapi.hf.space/--replicas/9rhs8/", token)

if len(audio) > 0:
    st.audio(audio.export().read())

    submit_button = st.button("Use this audio")

    if submit_button:
        # Audio recording must be converted first before feeding it to Whisper
        converted = np.array(audio.get_array_of_samples(), dtype=np.float32).reshape((-1, audio.channels))
        
        result = model.transcribe(converted)
        st.info("Transcribing...")
        
        st.success("Transcription complete")
        transcript = result['text']
        
        with st.expander("See transcript"):
            st.markdown(transcript)

        st.text("Sending image and request to the model. Please wait...")
        
        # Sample API call to LLavA 
        result_llava = client.predict(
            transcript,
            image,
            "Crop",
            fn_index=7
        )
        st.text("LLavA result: " + result_llava)
        
        # Text-to-Speech Translation
        tts_client = Client("https://facebook-seamless-m4t.hf.space/")
        tts_result = tts_client.predict(
            "T2ST (Text to Speech translation)",
            result_llava,  # Replacing "howdy" with LLavA result
            "english",
            "english"
        )
        st.text("Text-to-Speech Translation result: " + tts_result)
        
        
        # Download and autoplay the TTS audio
        st.text("Downloading and playing the translated audio:")
        audio_url = tts_result["audio_url"]
        st.audio(audio_url, format="audio/wav", start_time=0)
