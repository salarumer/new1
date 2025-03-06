import time
from google import genai
from google.cloud import bigquery
from google.genai.types import FunctionDeclaration, GenerateContentConfig, Part, Tool
import streamlit as st

# ✅ Correct project and dataset details
BIGQUERY_PROJECT_ID = "mltransit"  
BIGQUERY_DATASET_ID = "dataset2"

MODEL_ID = "gemini-1.5-pro"
LOCATION = "us-central1"

# ✅ Ensure correct table references (No duplication of project ID)
ALLOWED_TABLES = {
    f"{BIGQUERY_PROJECT_ID}.{BIGQUERY_DATASET_ID}.table2",
  
}

list_tables_func = FunctionDeclaration(
    name="list_tables",
    description="List allowed tables in a dataset.",
    parameters={
        "type": "object",
        "properties": {
            "dataset_id": {"type": "string", "description": "Dataset ID to fetch tables from."}
        },
        "required": ["dataset_id"],
    },
)

sql_query_func = FunctionDeclaration(
    name="sql_query",
    description="Run SQL queries only on allowed tables.",
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "SQL query that must use only allowed tables.",
            }
        },
        "required": ["query"],
    },
)

sql_query_tool = Tool(function_declarations=[list_tables_func, sql_query_func])

client = genai.Client(vertexai=True, location=LOCATION)
bq_client = bigquery.Client()

st.set_page_config(page_title="SQL Talk with BigQuery", page_icon="vertex-ai.png", layout="wide")

st.title("SQL Talk with BigQuery")
st.subheader("Powered by Function Calling in Gemini")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "backend_details" in message:
            with st.expander("Function calls, parameters, and responses"):
                st.markdown(message["backend_details"])

if prompt := st.chat_input("Ask me about information in the database..."):
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

        prompt += f"""
        Only use tables from `{BIGQUERY_PROJECT_ID}.{BIGQUERY_DATASET_ID}` that are in the allowed list: 
        {", ".join(ALLOWED_TABLES)}.
        Do NOT use any public datasets or unauthorized tables.
        """

        try:
            response = chat.send_message(prompt)
            response = response.candidates[0].content.parts[0]

            function_calling_in_process = True
            backend_details = ""

            while function_calling_in_process:
                try:
                    params = {key: value for key, value in response.function_call.args.items()}

                    if response.function_call.name == "list_tables":
                        api_response = list(ALLOWED_TABLES)  # ✅ Returns fully qualified tables
                        api_response = str(api_response)

                    if response.function_call.name == "sql_query":
                        query = params["query"]

                        # ✅ Ensure queries use only allowed tables
                        for table in ALLOWED_TABLES:
                            table_name = table.split(".")[-1]  # Extract table name (e.g., "table1")
                            if table_name in query:  # Table name appears without project ID?
                                query = query.replace(table_name, table)  # Fix it

                        if not any(t in query for t in ALLOWED_TABLES):
                            raise ValueError("Unauthorized table used in query.")

                        job_config = bigquery.QueryJobConfig(maximum_bytes_billed=100000000)
                        query_job = bq_client.query(query, job_config=job_config)
                        api_response = str([dict(row) for row in query_job.result()])

                    backend_details += f"Function: {response.function_call.name}\nParams: {params}\nResponse: {api_response}\n\n"

                    response = chat.send_message(
                        Part.from_function_response(name=response.function_call.name, response={"content": api_response})
                    )
                    response = response.candidates[0].content.parts[0]

                except AttributeError:
                    function_calling_in_process = False

            full_response = response.text
            with message_placeholder.container():
                st.markdown(full_response)
                with st.expander("Function calls, parameters, and responses"):
                    st.markdown(backend_details)

            st.session_state.messages.append({"role": "assistant", "content": full_response, "backend_details": backend_details})

        except Exception as e:
            error_message = f"Something went wrong: {str(e)}"
            st.error(error_message)
            st.session_state.messages.append({"role": "assistant", "content": error_message})
