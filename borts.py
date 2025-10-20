import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

# -------------------- APP TITLE --------------------
st.set_page_config(page_title="Dental AI Receptionist", page_icon="🦷", layout="centered")
st.title("🦷 Dental AI Receptionist Chatbot")
st.write("Hello! I’m your virtual dental assistant. How can I help you today?")

# -------------------- LOAD MODEL --------------------
@st.cache_resource
def load_model():
    model_name = "google/flan-t5-large"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

# -------------------- CHAT FUNCTION --------------------
def get_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=200, temperature=0.7, top_p=0.9)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# -------------------- UI --------------------
user_input = st.text_input("🧑 Patient:", placeholder="Type your question here...")

if user_input:
    dental_prompt = f"""
    You are a friendly and professional AI dental receptionist.
    Tasks:
    - Greet patients warmly.
    - Answer general dental questions (cleaning, braces, whitening, etc.)
    - Share clinic timings or booking info.
    - Stay polite and short.

    Patient: {user_input}
    AI Receptionist:
    """
    response = get_response(dental_prompt)
    st.markdown(f"**💬 AI Receptionist:** {response}")
