import streamlit as st
import tempfile
from pathlib import Path

from azure.identity import ClientSecretCredential
from azure.ai.agents import AgentsClient
from azure.ai.agents.models import FilePurpose, CodeInterpreterTool, MessageRole

# ----------------------------
# Azure configuration (from Streamlit secrets)
# ----------------------------
PROJECT_ENDPOINT = st.secrets["AZURE_OPENAI_ENDPOINT"]
TENANT_ID = st.secrets["AZURE_TENANT_ID"]
CLIENT_ID = st.secrets["AZURE_CLIENT_ID"]
CLIENT_SECRET = st.secrets["AZURE_CLIENT_SECRET"]

MODEL_DEPLOYMENT_NAME = "gpt-4o"  # adjust if needed

# ----------------------------
# Initialize Agent
# ----------------------------
def init_agent(uploaded_file):
    """Upload user file and create an agent."""

    # Save uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = Path(tmp.name)

    # Authenticate with Azure
    credential = ClientSecretCredential(
        tenant_id=TENANT_ID,
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
    )

    agent_client = AgentsClient(
        endpoint=PROJECT_ENDPOINT,
        credential=credential,
    )

    # Upload file
    uploaded_azure_file = agent_client.files.upload_and_poll(
        file_path=tmp_path,
        purpose=FilePurpose.AGENTS,
    )

    # Enable code interpreter
    code_tool = CodeInterpreterTool(file_ids=[uploaded_azure_file.id])

    # Create agent
    agent = agent_client.create_agent(
        model=MODEL_DEPLOYMENT_NAME,
        name="data-agent",
        instructions=(
            "You are an AI agent that analyzes the uploaded data file. "
            "Use Python when calculations or statistics are needed."
        ),
        tools=code_tool.definitions,
        tool_resources=code_tool.resources,
    )

    # Create conversation thread
    thread = agent_client.threads.create()

    return agent_client, agent, thread, tmp_path


# ----------------------------
# Streamlit App
# ----------------------------
def main():
    st.set_page_config(page_title="Azure AI Data Agent", layout="wide")
    st.title("📊 Azure AI Data Analysis Agent")

    # File uploader
    uploaded_file = st.file_uploader("Upload a data file (TXT / CSV)", type=["txt", "csv"])

    if uploaded_file and "agent_client" not in st.session_state:
        with st.spinner("Uploading file and creating agent..."):
            (
                st.session_state.agent_client,
                st.session_state.agent,
                st.session_state.thread,
                st.session_state.local_file_path,
            ) = init_agent(uploaded_file)
            st.session_state.messages = []
        st.success("✅ Agent is ready!")

    # Stop if agent not initialized
    if "agent_client" not in st.session_state:
        st.info("Please upload a file to start.")
        return

    # Show uploaded file content
    with st.expander("📂 Uploaded File Preview", expanded=True):
        with open(st.session_state.local_file_path, "r", encoding="utf-8", errors="ignore") as f:
            st.text(f.read())

    # Chat input
    user_input = st.chat_input("Ask a question about the data...")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})

        agent_client = st.session_state.agent_client
        agent = st.session_state.agent
        thread = st.session_state.thread

        # Send user message
        agent_client.messages.create(thread_id=thread.id, role="user", content=user_input)

        # Run agent
        run = agent_client.runs.create_and_process(thread_id=thread.id, agent_id=agent.id)

        if run.status == "failed":
            st.session_state.messages.append({"role": "agent", "content": f"❌ {run.last_error}"})
        else:
            reply = agent_client.messages.get_last_message_text_by_role(thread_id=thread.id, role=MessageRole.AGENT)
            if reply:
                st.session_state.messages.append({"role": "agent", "content": reply.text.value})

    # Render conversation
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Cleanup button
    if st.button("🧹 End Session"):
        st.session_state.agent_client.delete_agent(st.session_state.agent.id)
        st.session_state.clear()
        st.success("Session cleared.")


# ----------------------------
# Run app
# ----------------------------
if __name__ == "__main__":
    main()




