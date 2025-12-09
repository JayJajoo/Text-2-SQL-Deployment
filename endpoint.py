# ep.py
import sagemaker
from sagemaker.huggingface import HuggingFaceModel

# -------------------------------
# CONFIGURATION
# -------------------------------
sess = sagemaker.Session()
role = "arn:aws:iam::862763500155:role/SageMakerExecutionRole"
model_data = "s3://text-2-sql-buckt/my_model/model.tar.gz"              # Fixed: must be .tar.gz

# -------------------------------
# CREATE HUGGING FACE MODEL
# -------------------------------
huggingface_model = HuggingFaceModel(
    model_data=model_data,
    role=role,
    transformers_version="4.51",                                        # Can use shorthand
    pytorch_version="2.6",
    py_version="py312",
    entry_point="inference.py"  # Enabled for text2sql
)

# -------------------------------
# DEPLOY ENDPOINT ON GPU
# -------------------------------
predictor = huggingface_model.deploy(
    initial_instance_count=1,
    instance_type="ml.g5.xlarge",
    endpoint_name="text2sql-endpoint-2"
)

print("Endpoint deployed successfully!")
print("Endpoint name:", predictor.endpoint_name)