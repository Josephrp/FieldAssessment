import streamlit as st  
import numpy as np  
from audiorecorder import audiorecorder  
import whisper  
import os  
import streamlit.components.v1 as components  
from gradio_client import Client  
import tempfile  
import io  
import tempfile    


# Initialize your variables at the start of the script  
image = None  
audio = None  
transcript = None  
result_llava = None  
  
# Set the token as an environment variable  
os.environ["YOUR_API_TOKEN"] = "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"  
  
# Retrieve the token  
token = os.environ["YOUR_API_TOKEN"]  
  
# Define the Gradio server URL  
gradio_server_url = "https://teamtonic-llavaapi.hf.space/--replicas/kr8cw/"  
  
st.title("Assess This Picture")  
st.subheader("Take a picture first then ask a question & more!")  
  
image = st.camera_input("Camera input")  
  
# Write the uploaded file to a temporary file  
with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tf:  
    tf.write(image.read()) if image else None  # check if image is None
    temp_image_path = tf.name if image else None  # check if image is None
  
  
audio = audiorecorder("Click to record audio", "Click to stop recording")  
  
if len(audio) > 0:  
    audio_data = audio.export().read()  # Read audio data as bytes  
    audio_bytes_io = io.BytesIO(audio_data)  # Create a BytesIO object  
  
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as audio_file:  
        audio_file.write(audio_bytes_io.read())  # Write audio data to the temporary file  
        audio_file_path = audio_file.name  
  
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
        client = Client(src=gradio_server_url)  
        result_llava = client.predict(  
            transcript,  
            temp_image_path,  
            "Default",  
            fn_index=7  
        )  
  
        # Display LLavA result in a text box  
        st.subheader("LLavA Result:")  
        st.json(result_llava)   
  
    # Add a reset button to clear the interface  
    if st.button("Reset"):  
        st.experimental_rerun()  
