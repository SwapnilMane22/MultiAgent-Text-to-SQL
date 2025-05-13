#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import ollama
from ollama import Client
import subprocess
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
from sqlalchemy import create_engine, text, inspect
from typing import Dict, Any
import torch
import pymysql
import json
from typing import Dict, Any
import psutil
from GPUtil import getGPUs
from pathlib import Path


# In[ ]:


print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory/1e9:.2f} GB")
print(f"Compute Capability: {torch.cuda.get_device_capability()}")
print(f"Number of GPUs: {torch.cuda.device_count()}")
gpu = getGPUs()[0]
print(f"GPU Memory: {gpu.memoryUsed}")
print(f"Virtual Memory: {psutil.virtual_memory().percent}%")


# In[ ]:


try:
    subprocess.run(["ollama", "serve"], check=True, shell=True, timeout=5)
except:
    pass  # Already running in another terminal

# llama3.3:70b-instruct-q4_K_S
# sqlcoder:70b-alpha-q3_K_L

def verify_models():
    available_models = [m['name'] for m in ollama.list()['models']]
    required_models = [
        "sqlcoder:15b-q3_K_L",
        "phi4-reasoning:14b-plus-q4_K_M"
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
# Hardware Optimization Setup
# --------------------------
# def optimize_ollama_models():
#     # Configure Ollama to use GPU layers efficiently
#     ollama_model_config = {
#         "sqlcoder:15b-q3_K_L": {
#             "num_gpu": 18,  # Layers on GPU (fits 6GB VRAM)
#             "num_ctx": 768,  # Reduced context window
#             "num_threads": 4,  # CPU threads
#         },
#         "phi4-reasoning:14b-plus-q4_K_M": {
#             "num_gpu": 16,
#             "num_ctx": 768,
#             "num_threads": 4,
#         }
#     }

#     client = Client()

#     # Apply configurations
#     for model, config in ollama_model_config.items():
#         modelfile = f"""
#             FROM {model}
#             PARAMETER num_gpu {config['num_gpu']}
#             PARAMETER num_ctx {config['num_ctx']}
#             PARAMETER num_threads {config['num_threads']}
#             PARAMETER numa true
#         """

#         # Write Modelfile to disk
#         modelfile_path = f"{model.replace(':', '_')}.Modelfile"
#         with open(modelfile_path, "w") as f:
#             f.write(modelfile.strip())

#         client.create(
#             model=model,
#             from_=model
#         )

#         # Optionally, remove the Modelfile after creation
#         os.remove(modelfile_path)

# optimize_ollama_models()


# In[ ]:


# class MemoryAwareManager(CustomGroupChatManager):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.active_model = None

#     def switch_model(self, model_name):
#         """Unload current model and load new one"""
#         if self.active_model:
#             ollama.delete(self.active_model)
#         ollama.create(model_name)
#         self.active_model = model_name


# In[ ]:


# --------------------------
# Modified Agent Workflow
# --------------------------
def sequential_workflow(question_data):
    """Process one question at a time with memory cleanup"""
    manager = MemoryAwareManager(...)  # Initialize with previous config

    # Phase 1: Schema Mapping
    manager.switch_model("phi4-reasoning:14b-plus-q4_K_M")
    schema_result = agent1.generate_reply(
        f"Question: {question_data['question']}\nSchema: {SCHEMA}"
    )

    # Phase 2: SQL Generation
    manager.switch_model("sqlcoder:15b-q3_K_L")
    sql_query = agent2.generate_reply(schema_result)

    # Phase 3: Validation
    manager.switch_model("phi4-reasoning:14b-plus-q4_K_M")
    validation_result = agent3.validate_query(sql_query)

    return validation_result


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
        "model": "sqlcoder:15b-q3_K_L",
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama",
    },
    {  # Second model config
        "model": "phi4-reasoning:14b-plus-q4_K_M", 
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama",
    }
]
# Define LLM clients for each model
# llama3_client = OllamaChatCompletionClient(model="phi4-reasoning:14b-plus-q4_K_M")
# sqlcoder_client = OllamaChatCompletionClient(model="sqlcoder:15b-q3_K_L")


# # Define Multi-Agent architecture

# In[ ]:


class CustomGroupChatManager(GroupChatManager):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.accuracy_data = {"total": 0, "success": 0}

    def track_success(self, success: bool):
        self.accuracy_data["total"] += 1
        if success:
            self.accuracy_data["success"] += 1

    def get_accuracy(self):
        if self.accuracy_data["total"] == 0:
            return 0.0
        return (self.accuracy_data["success"] / self.accuracy_data["total"]) * 100


# In[ ]:


# Agent 1: Schema Mapping Agent
agent1 = AssistantAgent(
    name="Schema_Mapper",
    system_message="""You are a database schema expert. Analyze the input question and map it to the database schema.
    Available schema format for the current database:
    {schema}

    Output format must be JSON with tables as keys and values containing:
    - columns: list of relevant columns for the input
    - related_tables: list of tables needing joins
    """.format(
        schema_example=json.dumps({
            "account": {
                "columns": ["account_id", "district_id", "frequency", "date"],
                "related_tables": ["district", "disp", "loan", "order", "trans"]
            },
            "alignment": {
                "columns": ["id", "alignment"],
                "related_tables": ["superhero"]
            }
        }, indent=2),
        schema=json.dumps(schema, indent=2)
    ),
    llm_config={
        "config_list": [{
            "model": LLM_CONFIG[1]["model"],
            "base_url": LLM_CONFIG[1]["base_url"],
            "api_key": LLM_CONFIG[1]["api_key"]
        }]
    },
    max_consecutive_auto_reply=10,
)


# In[ ]:


# Agent 2: SQL Generation Agent
agent2 = AssistantAgent(
    name="SQL_Generator",
    system_message="""You are an expert SQL writer. Using the schema analysis from Schema_Mapper:
    1. Break down complex questions into subqueries
    2. Handle joins, nested queries, and aggregation
    3. Ensure MySQL syntax compliance
    4. Consider performance optimization
    5. Execute the sql query using the below function map and record the status of SQL query execution along with the  SQL query
    Return ONLY the SQL query without any explanation.""",
    llm_config={
        "config_list": [{
            "model": LLM_CONFIG[0]["model"],
            "base_url": LLM_CONFIG[0]["base_url"],
            "api_key": LLM_CONFIG[0]["api_key"]
        }]
    },
    max_consecutive_auto_reply=10,
)


# In[ ]:


# Agent 3: Validation & Feedback Agent
agent3 = UserProxyAgent(
    name="SQL_Validator",
    human_input_mode="NEVER",
    system_message="""Validate the output of SQL queries by comparing it with the input text and check if all the criteria has been satisfied along with successful execution for output SQL query. Provide RLAIF feedback:
    - If successful: "SUCCESS: [execution result]"
    - If failed: "ERROR: [error details]. Needed corrections: [specific fixes]"
    Maintain conversation history for iterative improvements.""",
    code_execution_config={"work_dir": "coding", "use_docker": False},
    llm_config={
        "config_list": [{
            "model": LLM_CONFIG[1]["model"],
            "base_url": LLM_CONFIG[1]["base_url"],
            "api_key": LLM_CONFIG[1]["api_key"]
        }]
    },
    function_map={"perform_query_on_mysql_databases": perform_query_on_mysql_databases},
)


# In[ ]:


# Agent 4: Group Manager
agent4 = CustomGroupChatManager(
    groupchat=GroupChat(
        agents=[agent1, agent2, agent3],
        messages=[],
        max_round=10,
        speaker_selection_method="round_robin",
    ),
    name="Group_Manager",
    system_message="Monitor SQL generation process. Ensure query execution accuracy >70%. Current execution accuracy: {accuracy}%",
    llm_config={
        "config_list": [{
            "model": LLM_CONFIG[1]["model"],
            "base_url": LLM_CONFIG[1]["base_url"],
            "api_key": LLM_CONFIG[1]["api_key"]
        }]
    },
)


# In[ ]:


# Register functions for execution
@agent3.register_for_execution()
@agent2.register_for_llm(description="Execute SQL query and return results")
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
def test_sql_generation(input_file: str, gold_file: str):
    with open(input_file) as f:
        test_cases = json.load(f)

    with open(gold_file) as f:
        gold_queries = f.read().split(';')

    for case in test_cases:
        print(f"\nProcessing case: {case['question']}")
        agent4.reset()
        agent1.reset()
        agent2.reset()
        agent3.reset()

        # Initiate conversation
        # Database: {case['db_id']}
        agent4.initiate_chat(
            agent1,
            message=f"""
            Question: {case['question']}
            Evidence: {case['evidence']}
            Goal: Generate executable SQL query for the given question
            """,
        )

        # Validate against gold standard
        generated_query = extract_last_query(agent4.chat_messages[agent3])
        gold_query = next(q for q in gold_queries if case['question'] in q)
        success = compare_queries(generated_query, gold_query)
        agent4.track_success(success)

        print(f"Accuracy: {agent4.get_accuracy():.2f}%")

def extract_last_query(messages: list) -> str:
    for msg in reversed(messages):
        if 'content' in msg and 'SELECT' in msg['content']:
            return msg['content'].split('```sql')[-1].split('```')[0].strip()
    return ""

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
        "./mini_dev/data_minidev/data_minidev/MINIDEV/mini_dev_mysql.json",
        "./mini_dev/data_minidev/data_minidev/MINIDEV/mini_dev_mysql_gold.sql"
    )
except Exception as e:
    print(f"Initialization failed: {e}")

