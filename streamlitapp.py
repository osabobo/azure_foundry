import os
import pandas as pd
import streamlit as st
from openai import AzureOpenAI

# ----------------------------
# Azure OpenAI configuration
# ----------------------------
AZURE_OPENAI_ENDPOINT = st.secrets["AZURE_OPENAI_ENDPOINT"]  # ✅ Correct
AZURE_OPENAI_KEY = st.secrets["AZURE_OPENAI_KEY"]
AZURE_OPENAI_API_VERSION = st.secrets.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
DEPLOYMENT_NAME = "gpt-4.1"  # your deployment name

# Initialize client
client = AzureOpenAI(
    api_key=AZURE_OPENAI_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version=AZURE_OPENAI_API_VERSION
)

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Azure OpenAI Data Chat", layout="wide")
st.title("📊 Azure OpenAI Data Chat App")

# Initialize conversation
if "messages" not in st.session_state:
    st.session_state.messages = []

# ----------------------------
# File upload
# ----------------------------
uploaded_file = st.file_uploader("Upload a CSV/TXT file to analyze", type=["csv", "txt"])
if uploaded_file:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file, sep="\t", header=None)
        st.session_state.df_preview = df.head(10)
        st.dataframe(st.session_state.df_preview)
    except Exception as e:
        st.error(f"Failed to read file: {e}")

# ----------------------------
# Show conversation
# ----------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ----------------------------
# Chat input
# ----------------------------
user_input = st.chat_input("Ask a question about your data or anything else...")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Build prompt
    if uploaded_file:
        prompt = f"You are a helpful data assistant. The user uploaded a dataset:\n{st.session_state.df_preview.to_csv(index=False)}\nNow answer their question:\n{user_input}"
    else:
        prompt = f"You are a helpful assistant. Answer the user question:\n{user_input}"

    try:
        response = client.chat.completions.create(
            model=DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            max_completion_tokens=1000,
            temperature=0.7,
        )
        reply = response.choices[0].message.content
        st.session_state.messages.append({"role": "assistant", "content": reply})

    except Exception as e:
        st.error(f"Error: {e}")

# ----------------------------
# Clear chat button
# ----------------------------
if st.button("🧹 Clear Chat"):
    st.session_state.messages = []
    if "df_preview" in st.session_state:
        del st.session_state.df_preview
    st.experimental_rerun()


