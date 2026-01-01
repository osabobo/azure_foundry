import streamlit as st
from pathlib import Path
from azure.identity import ClientSecretCredential
from azure.ai.agents import AgentsClient
from azure.ai.agents.models import FilePurpose, CodeInterpreterTool, MessageRole

# ----------------------------
# Streamlit secrets (Service Principal)
# ----------------------------
PROJECT_ENDPOINT = st.secrets["AZURE_API_ENDPOINT"]
TENANT_ID = st.secrets["AZURE_TENANT_ID"]
CLIENT_ID = st.secrets["AZURE_CLIENT_ID"]
CLIENT_SECRET = st.secrets["AZURE_CLIENT_SECRET"]

MODEL_DEPLOYMENT_NAME = "gpt-4o"

# ----------------------------
# Initialize Azure Agent
# ----------------------------
def init_agent():
    """Initialize the Azure Agent client and agent with the uploaded file."""
    script_dir = Path(__file__).parent
    file_path = script_dir / "data.txt"

    # Authenticate via Service Principal
    credential = ClientSecretCredential(
        tenant_id=TENANT_ID,
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
    )

    agent_client = AgentsClient(endpoint=PROJECT_ENDPOINT, credential=credential)

    # Upload file to Azure AI
    uploaded_file = agent_client.files.upload_and_poll(
        file_path=file_path,
        purpose=FilePurpose.AGENTS
    )

    code_interpreter = CodeInterpreterTool(file_ids=[uploaded_file.id])

    # Create Agent
    agent = agent_client.create_agent(
        model=MODEL_DEPLOYMENT_NAME,
        name="data-agent",
        instructions=(
            "You are an AI agent that analyzes the data in the uploaded file. "
            "Use Python to calculate statistical metrics as necessary."
        ),
        tools=code_interpreter.definitions,
        tool_resources=code_interpreter.resources,
    )

    # Create a conversation thread
    thread = agent_client.threads.create()

    return agent_client, agent, thread, file_path

# ----------------------------
# Main Streamlit App
# ----------------------------
def main():
    st.set_page_config(page_title="Azure Data Agent", layout="wide")
    st.title("📊 Azure AI Data Analysis Agent")

    # Initialize agent in session state
    if "agent_client" not in st.session_state:
        st.session_state.agent_client, st.session_state.agent, st.session_state.thread, st.session_state.file_path = init_agent()
        st.session_state.messages = []

    # Display file contents
    with st.expander("📂 Uploaded Data (data.txt)", expanded=True):
        with open(st.session_state.file_path, "r") as f:
            st.text(f.read())

    # Chat input
    user_input = st.chat_input("Ask a question about the data...")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})

        agent_client = st.session_state.agent_client
        agent = st.session_state.agent
        thread = st.session_state.thread

        # Send user message
        agent_client.messages.create(
            thread_id=thread.id,
            role="user",
            content=user_input,
        )

        # Run agent
        run = agent_client.runs.create_and_process(
            thread_id=thread.id,
            agent_id=agent.id
        )

        if run.status == "failed":
            st.session_state.messages.append(
                {"role": "agent", "content": f"Run failed: {run.last_error}"}
            )
        else:
            last_msg = agent_client.messages.get_last_message_text_by_role(
                thread_id=thread.id,
                role=MessageRole.AGENT
            )
            if last_msg:
                st.session_state.messages.append(
                    {"role": "agent", "content": last_msg.text.value}
                )

    # Render conversation
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # End session button
    if st.button("🧹 End Session and Delete Agent"):
        st.session_state.agent_client.delete_agent(st.session_state.agent.id)
        st.session_state.clear()
        st.success("Agent deleted and session cleared.")

# ----------------------------
# Run app
# ----------------------------
if __name__ == "__main__":
    main()


