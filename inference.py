# inference.py (NEW FILE - create this)
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

def model_fn(model_dir):
    """Load model and tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_dir,
        device_map="auto",
        torch_dtype=torch.float16
    )
    return {"model": model, "tokenizer": tokenizer}

def predict_fn(data, model_dict):
    """Run inference"""
    model = model_dict["model"]
    tokenizer = model_dict["tokenizer"]
    
    inputs = tokenizer(
        data["inputs"], 
        return_tensors="pt", 
        padding=True,
        truncation=True,
        max_length=512
    ).to(model.device)
    
    outputs = model.generate(
        **inputs, 
        max_length=512,
        num_beams=1,
        early_stopping=True
    )
    
    result = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    return {"generated_text": result[0]}