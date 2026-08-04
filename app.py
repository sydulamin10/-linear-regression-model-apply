import streamlit as st
import pickle

# Needed so pickle can find the model class
from model import LinearWeightModel  # noqa: F401

# Load trained linear regression model
with open("linear_model.pkl", "rb") as file:
    model = pickle.load(file)

st.set_page_config(page_title="AI Weight Predictor", page_icon="🤖")

st.title("🤖 AI Weight Predictor")

# Store chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! 👋 Enter your height in centimeters and I'll predict your weight."
        }
    ]

# Display previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Chat input
prompt = st.chat_input("Type your height (e.g. 170)")

if prompt:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.write(prompt)

    try:
        height = int(prompt)
        prediction = model.predict([[height]])
        weight = float(prediction[0][0])
        response = f"📏 Height: **{height} cm**\n\n⚖️ Predicted Weight: **{weight:.2f} kg**"

    except ValueError:
        response = "❌ Please enter a valid integer height (example: 170)."

    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )

    with st.chat_message("assistant"):
        st.markdown(response)
