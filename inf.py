# if.py
import sagemaker
from sagemaker.huggingface import HuggingFacePredictor

# -------------------------------
# CONFIGURATION
# -------------------------------
sess = sagemaker.Session()
endpoint_name = "text2sql-endpoint-2"

# -------------------------------
# CONNECT TO ENDPOINT
# -------------------------------
predictor = HuggingFacePredictor(endpoint_name=endpoint_name, sagemaker_session=sess)

# -------------------------------
# EXAMPLE INFERENCE WITH SCHEMA
# -------------------------------
schema = """
CREATE TABLE customers (
    customer_id INT PRIMARY KEY,
    first_name VARCHAR(50),
    last_name VARCHAR(50),
    email VARCHAR(100),
    city VARCHAR(50),
    state VARCHAR(2),
    created_at TIMESTAMP
);
"""

question = "Show all customers from New York."

# Format as text-to-sql models typically expect
input_text = f"Generate SQL for \n\ncontext: {schema}\n\n\query: {question}\nSQL:"

response = predictor.predict({"inputs": input_text})

print("Model output:")
print(response)