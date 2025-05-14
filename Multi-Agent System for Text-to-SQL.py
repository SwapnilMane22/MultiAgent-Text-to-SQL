#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import ollama
from ollama import Client
import subprocess
from subprocess import DEVNULL
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
from autogen.agentchat.agent import Agent
from sqlalchemy import create_engine, text, inspect
from typing import Dict, Any
import torch
import pymysql
import json
from typing import Dict, Any
import psutil
from GPUtil import getGPUs
from pathlib import Path
import sys
import io
import copy
import spacy

# In[ ]:

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory/1e9:.2f} GB")
print(f"Compute Capability: {torch.cuda.get_device_capability()}")
print(f"Number of GPUs: {torch.cuda.device_count()}")
gpu = getGPUs()[0]
print(f"GPU Memory: {gpu.memoryUsed}")
print(f"Virtual Memory: {psutil.virtual_memory().percent}%")


# In[ ]:

def start_ollama_server():
    """Start Ollama server with error-level logging"""
    env = os.environ.copy()
    # env["OLLAMA_LOG_LEVEL"] = "error"
    
    try:
        # Start Ollama with error logging
        process = subprocess.Popen(
            ["ollama", "serve"],
            env=env,
            stdout=DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
        )
        
        # Check for early errors
        _, stderr = process.communicate(timeout=5)
        if stderr:
            print(f"Ollama startup error: {stderr}")
            return None

        return process
    except FileNotFoundError:
        print("Ollama not installed or not in PATH")
        return None
    except subprocess.TimeoutExpired:
        # Server started successfully
        return process

# Usage example
ollama_process = start_ollama_server()

# llama3.3:70b-instruct-q4_K_S
# sqlcoder:70b-alpha-q3_K_L
# phi4-reasoning:14b-plus-q4_K_M
# sqlcoder:15b-q3_K_L
# sqlcoder:7b-q3_K_L
# phi4-mini-reasoning:3.8b-q4_K_M
# opencoder:8b-instruct-q4_K_M
# gemma3:27b-it-q4_K_M

model2 = "deepcoder:1.5b-preview-fp16"
model1 = "phi4-mini-reasoning:3.8b-q4_K_M"

def verify_models():
    available_models = [m['model'] for m in ollama.list()['models']]
    required_models = [
        model1,
        model2
    ]

    for model in required_models:
        if model not in available_models:
            print(f"Model {model} not found! Pulling...")
            ollama.pull(model)

verify_models()

def list_ollama_models():
    try:
        # Explicitly configure client for Windows
        client = Client(host='http://localhost:11434')

        # Verify connection
        # client.heartbeat()

        # Get models with error handling
        response = client.list()
        models = response.get('models', [])

        if not models:
            print("No models found. Install models using 'ollama pull <model_name>'")
            return

        print("\nAvailable Ollama Models:")
        print(f"{'Model Name':<40} {'Size':<15} {'Modified'}")
        print("-" * 70)
        for model in models:
            print(f"{model.get('model', 'N/A'):<40} "
                  f"{model.get('size', 0)/1e9:.2f} GB  "
                  f"{model.get('modified_at', 'N/A')}")

    except Exception as e:
        print(f"Error: {str(e)}")
        print("1. Ensure 'ollama serve' is running in a separate terminal")
        print("2. Check firewall settings allowing port 11434")
        print("3. Verify Ollama version with 'ollama --version'")

# Execute the function
list_ollama_models()


# # Setup MySQL functions

# In[ ]:


def connect_mysql():
    # Open database connection
    # Connect to the database"
    db = pymysql.connect(
        host="localhost",
        user="root",
        password="1234",
        database="BIRD",
        #unix_socket="/tmp/mysql.sock",
        port=3306,
    )
    return db

def execute_mysql_query(cursor, query):
    """Execute a MySQL query with error handling."""
    try:
        cursor.execute(query)
        # Only fetch results if it's a SELECT-like query
        if cursor.description:  # Checks if there are results to fetch
            return cursor.fetchall()
        return None  # For non-result queries like INSERT/UPDATE
    except pymysql.Error as e:
        print(f"[QUERY ERROR] {e.args[1]}")
        return None
    except Exception as e:
        print(f"[UNEXPECTED ERROR] During query execution: {e}")
        return None

def perform_query(query):
    """Execute query with connection safety and error handling."""
    db = None
    try:
        db = connect_mysql()
        cursor = db.cursor()
        result = execute_mysql_query(cursor, query)

        # Commit if needed (for write operations)
        if query.strip().lower().startswith(("insert", "update", "delete")):
            db.commit()

        return result
    except pymysql.Error as e:
        print(f"[DATABASE ERROR] Connection/execution failed: {e}")
        if db:  # Rollback if transaction exists
            db.rollback()
        return None
    finally:
        if db:
            db.close()

def get_bird_schema():
    """
    Retrieves the schema of the BIRD database, including tables, columns, and related tables.

    Returns:
        A dictionary where each key is a table name, and the value is another dictionary
        containing 'columns' (list of column names and types) and 'related_tables' (list of related tables).
    """
    # Query to get all tables in the BIRD database
    table_query = """
        SELECT TABLE_NAME 
        FROM INFORMATION_SCHEMA.TABLES 
        WHERE TABLE_SCHEMA = 'BIRD';
    """
    tables_result = perform_query(table_query)
    tables = [row[0] for row in tables_result]

    schema = {}

    # Retrieve columns for each table
    for table in tables:
        column_query = f"""
            SELECT COLUMN_NAME, DATA_TYPE 
            FROM INFORMATION_SCHEMA.COLUMNS 
            WHERE TABLE_SCHEMA = 'BIRD' AND TABLE_NAME = '{table}';
        """
        columns_result = perform_query(column_query)
        columns = [{'name': col[0], 'type': col[1]} for col in columns_result]
        schema[table] = {
            'columns': columns,
            'related_tables': []
        }

    # Retrieve foreign key relationships within the BIRD database
    fk_query = """
        SELECT DISTINCT TABLE_NAME, REFERENCED_TABLE_NAME 
        FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE 
        WHERE TABLE_SCHEMA = 'BIRD' 
            AND REFERENCED_TABLE_SCHEMA = 'BIRD' 
            AND REFERENCED_TABLE_NAME IS NOT NULL;
    """
    fk_result = perform_query(fk_query)

    # Populate related_tables (both directions)
    for source_table, related_table in fk_result:
        # Add related table to the source table's list
        if source_table in schema and related_table not in schema[source_table]['related_tables']:
            schema[source_table]['related_tables'].append(related_table)
        # Add source table to the related table's list
        if related_table in schema and source_table not in schema[related_table]['related_tables']:
            schema[related_table]['related_tables'].append(source_table)

    return schema

def perform_query_on_mysql_databases(query: str) -> str:
    """User-facing query executor with string conversion"""
    result = perform_query(query)
    return str(result) if result else "ERROR"


# # Test MySQL Setup

# In[ ]:


# Quick test connection
try:
    conn = connect_mysql()
    print("Successfully connected to MySQL!")
    conn.close()
except Exception as e:
    print(f"Connection failed: {e}")

schema = get_bird_schema()
# for table, details in schema.items():
#     print(f"Table: {table}")
#     print(f"Columns: {[col['name'] for col in details['columns']]}")
#     print(f"Related Tables: {details['related_tables']}\n")

### Syntax to execute the query ###
# query = """ """
# result = perform_query(query)


# # LLM Configuration

# In[ ]:

# --------------------------
# Resource Monitoring
# --------------------------
def resource_monitor():
    """Check available resources before each operation"""
    if gpu.memoryUsed > 5500:  # 6GB VRAM
        raise MemoryError("GPU memory exhausted")

    if psutil.virtual_memory().percent > 90:
        raise MemoryError("System memory exhausted")

# --------------------------
# Modified Agent Classes
# --------------------------
# class SafeAssistantAgent(AssistantAgent):
#     def generate_reply(self, *args, **kwargs):
#         resource_monitor()
#         return super().generate_reply(*args, **kwargs)


# In[ ]:


# Configuration for Ollama models
LLM_CONFIG = [
    {  # First model config
        "model": model1, # #sqlcoder:7b-q3_K_L
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama",
        "timeout": 500,
        "price": [0.0, 0.0],
    },
    {  # Second model config
        "model": model2, 
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama",
        "timeout": 500,
        "price": [0.0, 0.0],
    }
]
# Define LLM clients for each model
# llama3_client = OllamaChatCompletionClient(model="phi4-mini-reasoning:3.8b-q4_K_M")
# sqlcoder_client = OllamaChatCompletionClient(model="sqlcoder:7b-q3_K_L")


# # Define Multi-Agent architecture

# In[ ]:

# Load spaCy NER model
nlp = spacy.load("en_core_web_sm")


# --- NER Extraction ---
def extract_keywords(question):
    doc = nlp(question)
    # Extract named entities + noun chunks
    keywords = set(ent.text.lower() for ent in doc.ents)
    keywords.update(chunk.root.text.lower() for chunk in doc.noun_chunks)
    return list(keywords)

# --- Schema Matcher ---
def match_schema(keywords, schema_dict):
    matched = {}
    for table, columns in schema_dict.items():
        table_lower = table.lower()
        match_found = any(kw in table_lower for kw in keywords)
        matched_columns = [col for col in columns if any(kw in col.lower() for kw in keywords)]

        if match_found or matched_columns:
            matched[table] = matched_columns if matched_columns else columns
    return matched

# --- Prompt Builder ---
def build_prompt(question, matched_schema):
    schema_str = "\n".join([f"- {tbl}({', '.join(cols)})" for tbl, cols in matched_schema.items()])
    return f"""
    You are a database schema expert.

    Given a Question: {question}, below schema, you must identify and match relevant entities (such as table names and column names) and explain the question to provide similarity with the provided database schema.

    We want to specifically use ONLY the below provided schema to derive the required table columns and relationships:

    Schema:
    {json.dumps(schema_str, indent=2)}


    Fetched Schema:

    {json.dumps(schema_str, indent=2)}

    Steps to follow:
    1. Explain the question to provide similarity with the provided database schema. Identify the entities in the question and determine which category they would belong to and then breakdown the question into sub question w.r.t. the schema.
    2. Match these entities with relevant table names and column names specifically from the provided schema ONLY. Additionally, if you're not able to explain the {question} then ONLY look up at entire database schema ONLY to avoid halucination using the SCHEMA: {json.dumps(schema, indent=2)}
    3. Identify relationships using 'related_tables' to determine necessary joins.
    4. Summarise which all tables are needed along with it's column names fetched directly from the schema in the json format.

    The output should be summarised json format with tables, columns and related tables along with the {question}.

    Strictly, Do NOT explain anything. Only return output JSON (with tables, columns, and related tables), question. Pass this output json to {{MySQL_Database_Engineer}}.
    """


# In[ ]:


# Agent 2: SQL Generation Agent
agent2 = AssistantAgent(
    name="MySQL_Database_Engineer",
    system_message="""You are responsible for validating SQL query outputs by comparing them with the input question. Your goals are:
    1. Take the input from {Database_Schema_Mapper} and parse the output {SQL Query}
    2. Ensure the generated SQL query meets all requirements stated in the question and is supported by the evidence.
    3. Attempt to execute the query and validate its correctness based on the execution result.
    4. Write the final MySQL query
    5. Make sure the sql query is executable
    6. Return ONLY the MySQL query without any explanation that would provide output for the input question.

    Provide RLAIF-style feedback as follows:
    - If the query is correct and executes successfully: 
      "SUCCESS: [brief description of execution result]"

    - If the query is incorrect or execution fails: 
      "ERROR: [error details]. Needed corrections: [specific and actionable fixes]"

    Maintain the conversation history to support iterative improvements. 
    Strictly, Do NOT explain anything, just return the output MySQL query. Pass this MySQL query to {SQL_Validator}. If the output is empty, circle back to {Database_Schema_Mapper}
    """,
    llm_config={
        "config_list": [{
            "model": LLM_CONFIG[1]["model"],
            "base_url": LLM_CONFIG[1]["base_url"],
            "api_key": LLM_CONFIG[1]["api_key"],
            "price": LLM_CONFIG[1]["price"]
        }],
        "timeout": LLM_CONFIG[1]["timeout"]
    },
    function_map={"perform_query_on_mysql_databases": perform_query_on_mysql_databases},
    max_consecutive_auto_reply=10,
)


# In[ ]:


# Agent 3: Validation & Feedback Agent
agent3 = UserProxyAgent(
    name="SQL_Validator",
    human_input_mode="NEVER",
    system_message="""
    You are responsible for validating SQL query outputs by comparing them with the input question and provided evidence. Your goals are:
    1. Take the input from {MySQL_Database_Engineer} and parse the output {SQL Query}
    2. Ensure the generated SQL query meets all requirements stated in the question and is supported by the evidence.
    3. Attempt to execute the query and validate its correctness based on the execution result.

    Provide RLAIF-style feedback as follows:
    - If the query is correct and executes successfully: 
      "SUCCESS: [brief description of execution result]"

    - If the query is incorrect or execution fails: 
      "ERROR: [error details]. Needed corrections: [specific and actionable fixes]"

    Maintain the conversation history to support iterative improvements. 
    Also, broadcast the final validation status (success or failure) to the entire group. In the case of failure, share corrections needed for refinement using RLAIF style.
    """,
    code_execution_config={
        "work_dir": "coding", 
        "use_docker": False,
        "timeout": 10000  # Execution timeout
    },
    llm_config={
        "config_list": [{
            "model": LLM_CONFIG[0]["model"],
            "base_url": LLM_CONFIG[0]["base_url"],
            "api_key": LLM_CONFIG[0]["api_key"],
            "price": LLM_CONFIG[0]["price"]
        }],
        "timeout": LLM_CONFIG[0]["timeout"]
    },
    function_map={"perform_query_on_mysql_databases": perform_query_on_mysql_databases},
)


# In[ ]:


# agent4.groupchat.speaker_selection_method = agent4.select_speaker

# In[ ]:


# Register functions for execution
@agent3.register_for_execution()
# @agent2.register_for_llm(description="Execute SQL query and return results")
def perform_query_on_mysql(query: str) -> str:
    """Execute SQL query and return results as string"""
    result = perform_query(query)
    return str(result) if result else "ERROR"


# # Test the Pipeline

# In[ ]:


# Windows path handling for data files
def get_windows_path(relative_path):
    return str(Path(os.getcwd()) / relative_path)

# Test pipeline
def test_sql_generation(input_file: str):
    with open(input_file) as f:
        original_test_cases = json.load(f)

    
    # Extract only the first 10 input test cases
    # first_10_inputs = [tc["input"] for tc in original_test_cases[:10]]
    total_cases = len(original_test_cases[:10])
    passed_cases = 0
    updated_test_cases = []
    output_file = "./output_minidev_results.json"

    for idx, case in enumerate(original_test_cases[:10], 1):
        print(f"\nProcessing case {idx}/{total_cases}: {case['question']}")
        agent2.clear_history()
        agent3.clear_history()
        # agent4.groupchat.messages.clear()

        updated_case = copy.deepcopy(case)

        try:
            keywords = extract_keywords(case['question'])
            matched_schema = match_schema(keywords, schema)
            system_prompt = build_prompt(case['question'], matched_schema)
            # Agent 1: Schema Mapping Agent
            agent1 = AssistantAgent(
                name="Database_Schema_Mapper",
                system_message=system_prompt,
                llm_config={
                    "config_list": [{
                        "model": LLM_CONFIG[0]["model"],
                        "base_url": LLM_CONFIG[0]["base_url"],
                        "api_key": LLM_CONFIG[0]["api_key"],
                        "price": LLM_CONFIG[0]["price"]
                    }],
                    "timeout": LLM_CONFIG[0]["timeout"]
                },
                max_consecutive_auto_reply=10,
            )

            
            # Agent 4: Group Manager
            agent4 = GroupChatManager(
                groupchat=GroupChat(
                    agents=[agent1, agent2, agent3],
                    messages=[],
                    max_round=15,
                    speaker_selection_method="round_robin",
                    allow_repeat_speaker=[agent2] 
                ),
                name="Group_Manager",
                system_message="Monitor SQL generation process. Count how many cases i.e. questions you've passed successfully and account for failures. Also provide Current execution accuracy: {accuracy}%",
                llm_config={
                    "config_list": [{
                        "model": LLM_CONFIG[0]["model"],
                        "base_url": LLM_CONFIG[0]["base_url"],
                        "api_key": LLM_CONFIG[0]["api_key"],
                        "price": LLM_CONFIG[0]["price"]
                    }],
                    "timeout": LLM_CONFIG[0]["timeout"]
                },
            )
            agent4.groupchat.messages.clear()

            agent4.initiate_chat(
                recipient=agent1,
                # Schema: {json.dumps(schema, indent=2)}
                # Goal: Generate executable MySQL query for the given question
                message=f"""
                Schema: {json.dumps(schema, indent=2)}
                Question: {case['question']}
                """,
                participants=[agent1, agent2, agent3]
            )


            # print("=== Agent 1: MySQL_Database_Engineer ===")
            # print("System Message:", agent1.system_message)

            # print("Agent 1 output:", extract_last_query(agent4.chat_messages.get(agent1, [])))

            # print("=== Agent 2: MySQL_Database_Engineer ===")
            # print("System Message:", agent2.system_message)
            # print("Generated SQL Query:", extract_last_query(agent4.chat_messages.get(agent2, [])))

            # print("=== Agent 3: SQL_Validator ===")
            # print("System Message:", agent3.system_message)
            # print("Validation Result:", agent4.chat_messages.get(agent3, []))



            if agent4.groupchat.is_terminated:
                print(f"⚠️ Max round limit reached for question: '{case['question']}'")
                updated_test_cases.append(updated_case)
                continue

            if "SUCCESS" in agent4.chat_messages.get(agent3, []):
                generated_query = extract_last_query(agent4.chat_messages.get(agent2, []))
                gold_query = case.get("SQL", "").strip()
                if not gold_query:
                    print("⚠️ SQL field not found in test case, skipping.")
                    updated_test_cases.append(updated_case)
                    continue

                success = compare_queries(generated_query, gold_query)
                agent4.track_success(success)

                if success:
                    passed_cases += 1
                    updated_case["output_MySQL"] = generated_query
                else:
                    print("❌ Query mismatch or execution failure.")

        except Exception as e:
            print(f"❌ Error processing question: {case['question']}\nError: {str(e)}")

        updated_test_cases.append(updated_case)

        try:
            with open(output_file, "w") as outf:
                json.dump(updated_test_cases, outf, indent=2)
        except Exception as write_err:
            print(f"⚠️ Failed to write to file: {str(write_err)}")

        print(f"✅ Current Accuracy: {(passed_cases / idx) * 100:.2f}%")

    print(f"\n🎯 Final Accuracy: {(passed_cases / total_cases) * 100:.2f}% ({passed_cases}/{total_cases})")

def extract_last_query(messages: list) -> str:
    for msg in reversed(messages):
        if msg['name'] == agent2.name:
            content = msg.get('content', '')
            if '```sql' in content:
                return content.split('```sql')[-1].split('```')[0].strip()
            return content.strip()
    return ""
    #     if 'content' in msg and 'SELECT' in msg['content']:
    #         return msg['content'].split('```sql')[-1].split('```')[0].strip()
    # return ""

def compare_queries(generated: str, gold: str) -> bool:
    def normalize_query(query):
        return " ".join(query.lower().split()).replace(";", "").strip()
    return normalize_query(generated) == normalize_query(gold)

# In[ ]:


try:
    # Verify MySQL connection
    conn = connect_mysql()
    conn.close()
    resource_monitor()

    # Run test pipeline
    test_sql_generation(
        "./mini_dev/data_minidev/data_minidev/MINIDEV/mini_dev_mysql.json"
    )
except Exception as e:
    print(f"Initialization failed: {e}")