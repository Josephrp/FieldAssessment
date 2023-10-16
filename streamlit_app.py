import streamlit as st  
import gradio as gr  
import numpy as np      
from audiorecorder import audiorecorder      
import whisper      
import os      
import streamlit.components.v1 as components   
import tempfile      
import io      
import requests    
import json  
  
class MyClient:          
    def __init__(self, server_url):      
        self.src = server_url      
      
    def _make_request(self, transcript, image_path, preprocess_type, fn_index):          
        data = {    
            'transcript': transcript,    
            'image_path': image_path,    
            'preprocess_type': preprocess_type,    
            'fn_index': fn_index,    
        }  
        headers = {'Content-Type': 'application/json'}          
        response = requests.post(self.src, headers=headers, json=data)          
        return response.json()  
  
    def predict(self, transcript, image_path, preprocess_type, fn_index):          
        return self._make_request(transcript, image_path, preprocess_type, fn_index)  

  
image = None      
audio = None      
transcript = None      
result_llava = None  
submit_button = None
result = None
  
os.environ["YOUR_API_TOKEN"] = "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"  
token = os.environ["YOUR_API_TOKEN"]  
gradio_server_url = "https://teamtonic-llavaapi.hf.space/--replicas/z8hfj/"    
client = MyClient(gradio_server_url)    
  
st.title("Field Assessment")    
st.subheader("Take a picture first then ask a question to assess in the field!")    
    
image = st.camera_input("Camera input")    
    
with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tf:    
    tf.write(image.read()) if image else None    
    temp_image_path = tf.name if image else None    
  
audio = audiorecorder("Click to record audio", "Click to stop recording")    
    
if len(audio) > 0:    
    audio_data = audio.export().read()    
    audio_bytes_io = io.BytesIO(audio_data)    
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as audio_file:    
        audio_file.write(audio_bytes_io.read())    
        audio_file_path = audio_file.name    
    st.audio(audio_file_path, format="audio/wav")    
    submit_button = st.button("Use this audio")    
    
if submit_button:  
    model = whisper.load_model("base")  
    result = model.transcribe(audio_file_path)  
    st.info("Transcribing...")  
    st.success("Transcription complete")  
    transcript = result['text']  
  
    with st.expander("See transcript"):  
        st.markdown(transcript)  
        # Create a JSON object  
        data = {  
            "transcript": transcript,  
            "image_path": temp_image_path,   
            "preprocess_type": "Resize",  
            "fn_index": 7  
        }  
        st.text("Sending image and request to the model. Please wait...")  
        result_llava = client.predict(transcript, temp_image_path, "Resize", 7)  
        st.subheader("LLavA Result:")  
        st.json(result_llava)  

print(result)    
if st.button("Reset"):
    image = None        
    audio = None        
    transcript = None        
    result_llava = None    
    submit_button = None  
    result = None  
    st.experimental_rerun()  
    st.experimental_rerun()  
