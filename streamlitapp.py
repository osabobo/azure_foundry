import streamlit as st
from azure.identity import ClientSecretCredential
from azure.ai.agents import AgentsClient
from azure.ai.agents.models import (
    FilePurpose,
    CodeInterpreterTool,
    MessageRole,
)

# ----------------------------
# Config
# ----------------------------
st.set_page_config(
    page_title="Azure AI Data Agent",
    page_icon="📊",
    layout="wide",
)

PROJECT_ENDPOINT = st.secrets["AZURE_PROJECT_ENDPOINT"]
TENANT_ID = st.secrets["AZURE_TENANT_ID"]
CLIENT_ID = st.secrets["AZURE_CLIENT_ID"]
CLIENT_SECRET = st.secrets["AZURE_CLIENT_SECRET"]

MODEL = "gpt-4o"

# ----------------------------
# Azure Client (Cached)
# ----------------------------
@st.cache_resource
def get_agent_client():
    credential = ClientSecretCredential(
        tenant_id=TENANT_ID,
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
    )
    return AgentsClient(
        endpoint=PROJECT_ENDPOINT,
        credential=credential,
    )

# ----------------------------
# UI
# ----------------------------
st.title("📊 Azure AI Data Analysis Agent")
st.caption("Secure • Serverless • Production Ready")

uploaded_file = st.file_uploader(
    "Upload a CSV or TXT file",
    type=["csv", "txt"],
)

# ----------------------------
# Agent Setup
# ----------------------------
if uploaded_file and "agent" not in st.session_state:
    with st.spinner("Creating secure agent..."):
        client = get_agent_client()

        # Upload directly from memory (NO DISK)
        uploaded = client.files.upload_and_poll(
            file_bytes=uploaded_file.getvalue(),
            file_name=uploaded_file.name,
            purpose=FilePurpose.AGENTS,
        )

        tool = CodeInterpreterTool(
            file_ids=[uploaded.id]
        )

        agent = client.create_agent(
            name="data-agent",
            model=MODEL,
            instructions=(
                "You analyze uploaded datasets. "
                "Use Python for calculations and statistics."
            ),
            tools=tool.definitions,
            tool_resources=tool.resources,
        )

        thread = client.threads.create()

        st.session_state.client = client
        st.session_state.agent = agent
        st.session_state.thread = thread
        st.session_state.messages = []

    st.success("Agent ready")

# ----------------------------
# Chat
# ----------------------------
if "agent" in st.session_state:
    prompt = st.chat_input("Ask a question about your data")

    if prompt:
        st.session_state.messages.append(
            {"role": "user", "content": prompt}
        )

        client = st.session_state.client
        agent = st.session_state.agent
        thread = st.session_state.thread

        client.messages.create(
            thread_id=thread.id,
            role="user",
            content=prompt,
        )

        run = client.runs.create_and_process(
            thread_id=thread.id,
            agent_id=agent.id,
        )

        if run.status == "failed":
            response = f"❌ {run.last_error}"
        else:
            msg = client.messages.get_last_message_text_by_role(
                thread_id=thread.id,
                role=MessageRole.AGENT,
            )
            response = msg.text.value if msg else "No response"

        st.session_state.messages.append(
            {"role": "assistant", "content": response}
        )

# ----------------------------
# Render Messages
# ----------------------------
for m in st.session_state.get("messages", []):
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# ----------------------------
# Cleanup
# ----------------------------
if st.button("🧹 End Session"):
    try:
        st.session_state.client.delete_agent(
            st.session_state.agent.id
        )
    except Exception:
        pass
    st.session_state.clear()
    st.success("Session cleared")



