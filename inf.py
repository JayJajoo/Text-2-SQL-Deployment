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
schema = """CREATE TABLE tour_revenue (tour_id INT, operator_id INT, revenue INT, tour_type VARCHAR); CREATE VIEW virtual_tour_revenue AS SELECT * FROM tour_revenue WHERE tour_type = 'Virtual'; CREATE TABLE japan_tourism (tourist_id INT, country VARCHAR); CREATE VIEW japan_virtual_tours AS SELECT * FROM virtual_tour_revenue JOIN japan_tourism ON virtual_tour_revenue.tour_id = japan_tourism.tourist_id WHERE japan_tourism.country = 'Japan';"""

question = "What is the total revenue generated from virtual tour experiences in Japan?"

# Format as text-to-sql models typically expect
input_text = f"context: {schema} query: {question}"

response = predictor.predict({"inputs": input_text})

print("Model output:")
print(response)