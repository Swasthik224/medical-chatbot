import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# ✅ Use valid model
model_name = "microsoft/biogpt"

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

st.title("🩺 AI Medical Chatbot")

user_input = st.text_input("Enter your symptoms:")

if st.button("Ask") and user_input:
    inputs = tokenizer(user_input, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    st.write("### 🤖 Response:")
    st.write(response)

    st.warning("This is not medical advice. Consult a doctor.")
