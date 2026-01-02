import streamlit as st
from azure.identity import ClientSecretCredential
from azure.ai.agents import AgentsClient
from azure.ai.agents.models import FilePurpose, MessageRole

# ----------------------------
# Azure Configuration (from Streamlit secrets)
# ----------------------------
PROJECT_ENDPOINT = st.secrets["AZURE_API_ENDPOINT"]
TENANT_ID = st.secrets["AZURE_TENANT_ID"]
CLIENT_ID = st.secrets["AZURE_CLIENT_ID"]
CLIENT_SECRET = st.secrets["AZURE_CLIENT_SECRET"]

AGENT_ID = st.secrets["AZURE_AGENT_ID"]  # Use your existing agent

# Authenticate once
credential = ClientSecretCredential(
    tenant_id=TENANT_ID,
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
)

agent_client = AgentsClient(
    endpoint=PROJECT_ENDPOINT,
    credential=credential,
)

# ----------------------------
# Streamlit App
# ----------------------------
st.set_page_config(page_title="Azure AI Agent (Streamlit Cloud)", layout="wide")
st.title("📊 Azure AI Agent (Direct Cloud Upload)")

# File uploader
uploaded_file = st.file_uploader("Upload a data file (TXT / CSV)", type=["txt", "csv"])

if uploaded_file:
    with st.spinner("Uploading file to Azure Foundry..."):
        # Upload directly from uploaded_file (Streamlit cloud file-like object)
        azure_file = agent_client.files.upload_and_poll(
            file_path=uploaded_file,
            purpose=FilePurpose.AGENTS,
        )

        st.success(f"File uploaded! File ID: {azure_file.id}")

    # Ask agent
    user_input = st.text_area("Ask the agent about your data:")

    if st.button("Ask Agent"):
        if user_input.strip() == "":
            st.warning("Please enter a question.")
        else:
            # Create a conversation thread
            thread = agent_client.threads.create()

            # Send user message with file reference
            agent_client.messages.create(
                thread_id=thread.id,
                role="user",
                content=f"{user_input}\n\nFile ID: {azure_file.id}"
            )

            # Run the existing agent
            run = agent_client.runs.create_and_process(
                thread_id=thread.id,
                agent_id=AGENT_ID,
            )

            if run.status == "failed":
                st.error(f"Agent failed: {run.last_error}")
            else:
                # Fetch agent reply
                reply = agent_client.messages.get_last_message_text_by_role(
                    thread_id=thread.id,
                    role=MessageRole.AGENT,
                )
                if reply:
                    st.markdown("**Agent Response:**")
                    st.write(reply.text.value)




