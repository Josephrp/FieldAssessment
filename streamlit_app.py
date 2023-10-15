import streamlit as st
import numpy as np
from audiorecorder import audiorecorder
from gradio_client import Client
import whisper

import os

# Set the token as an environment variable
os.environ["YOUR_API_TOKEN"] = "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"

# Retrieve the token
token = os.environ["YOUR_API_TOKEN"]

st.title("Team Tonic Demo")
st.subheader("Take a picture first and speak your request for the model")

image = st.camera_input("Camera input")

audio = audiorecorder("Click to record audio", "Click to stop recording")

model = whisper.load_model("base")

client = Client("https://tonic-llava-v1.hf.space/--replicas/qj67l/", token)

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
        result = client.predict(
            transcript,
            image,
            "Crop",
            fn_index=7
        )
        print(result)
       # The 'result' variable holds the response to use for Text-to-Speech  
