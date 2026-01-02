from azure.identity import ClientSecretCredential
from openai import AzureOpenAI

credential = ClientSecretCredential(
    tenant_id=st.secrets["AZURE_TENANT_ID"],
    client_id=st.secrets["AZURE_CLIENT_ID"],
    client_secret=st.secrets["AZURE_CLIENT_SECRET"],
)

client = AzureOpenAI(
    azure_endpoint=st.secrets["AZURE_OPENAI_ENDPOINT"],
    api_version=st.secrets["AZURE_OPENAI_API_VERSION"],
    azure_ad_token_provider=credential
)

models = client.models.list()
st.write(models)




