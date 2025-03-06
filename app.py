# pylint: disable=broad-exception-caught,invalid-name

import time

from google import genai
from google.cloud import bigquery
from google.genai.types import FunctionDeclaration, GenerateContentConfig, Part, Tool
import streamlit as st

BIGQUERY_DATASET_ID = "dataset2"
TABLE_ID = "table2"  # Restricting to table2 only
MODEL_ID = "gemini-1.5-pro"
LOCATION = "us-central1"

get_table_func = FunctionDeclaration(
    name="get_table",
    description="Get information about table2, including the description, schema, and number of rows that will help answer the user's question.",
    parameters={
        "type": "object",
        "properties": {
            "table_id": {
                "type": "string",
                "description": "Fully qualified ID of table2.",
            }
        },
        "required": [
            "table_id",
        ],
    },
)

sql_query_func = FunctionDeclaration(
    name="sql_query",
    description="Get information from table2 in BigQuery using SQL queries.",
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "SQL query on a single line for table2. Always use the fully qualified dataset and table names.",
            }
        },
        "required": [
            "query",
        ],
    },
)

sql_query_tool = Tool(
    function_declarations=[get_table_func, sql_query_func],
)

client = genai.Client(vertexai=True, location=LOCATION)

st.set_page_config(
    page_title="SQL Talk with BigQuery",
    page_icon="vertex-ai.png",
    layout="wide",
)

st.title("SQL Talk with BigQuery")
st.subheader("Restricted to table2")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"].replace("$", r"\$"))  # noqa: W605

if prompt := st.chat_input("Ask me about table2..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        chat = client.chats.create(
            model=MODEL_ID,
            config=GenerateContentConfig(temperature=0, tools=[sql_query_tool]),
        )
        bq_client = bigquery.Client()

        prompt += f"""
            Only use table2 in dataset2. Do not access any other table.
        """
        
        try:
            response = chat.send_message(prompt)
            response = response.candidates[0].content.parts[0]
            
            api_requests_and_responses = []
            backend_details = ""

            if response.function_call.name == "get_table":
                api_response = bq_client.get_table(f"{BIGQUERY_DATASET_ID}.{TABLE_ID}")
                api_response = api_response.to_api_repr()
                api_requests_and_responses.append([
                    response.function_call.name, {}, str(api_response)
                ])
            
            if response.function_call.name == "sql_query":
                query = response.function_call.args["query"]
                if TABLE_ID not in query:
                    raise Exception("Query must only use table2")
                
                job_config = bigquery.QueryJobConfig(maximum_bytes_billed=100000000)
                query_job = bq_client.query(query, job_config=job_config)
                api_response = str([dict(row) for row in query_job.result()])
                api_requests_and_responses.append([
                    response.function_call.name, {"query": query}, api_response
                ])

            full_response = api_response
            with message_placeholder.container():
                st.markdown(full_response.replace("$", r"\$"))  # noqa: W605
            
            st.session_state.messages.append(
                {"role": "assistant", "content": full_response}
            )
        except Exception as e:
            error_message = f"Error: {str(e)}"
            st.error(error_message)
            st.session_state.messages.append({"role": "assistant", "content": error_message})
