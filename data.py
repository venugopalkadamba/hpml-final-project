import pickle
from huggingface_hub import login
from datasets import load_dataset
from transformers import AutoTokenizer


login("")

model_name = "meta-llama/Llama-3.2-1B"
# model_name = "meta-llama/Llama-2-7b-chat-hf"

tokenizer = AutoTokenizer.from_pretrained(model_name)

data = load_dataset("stas/openwebtext-10k")

def tokenize(example):
  return tokenizer(example["text"])

tokenized_data = data["train"].map(tokenize)
data_as_dict = tokenized_data.to_dict()

pickle.dump(data_as_dict, open("data_llama3.pkl", "wb"))