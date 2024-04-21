from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification

### Pipelines makes it super easy to apply any NLP task
model_name = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

### Create pipeline object passing in sentiment-analysis task, model, and tokenizer
classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
sentiment = classifier("I Love HuggingFace")

print(sentiment)