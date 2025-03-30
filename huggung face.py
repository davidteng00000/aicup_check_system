from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "taide/TAIDE-LX-7B-Chat"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

