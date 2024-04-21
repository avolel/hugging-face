from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification

### A Tokenizer converts text into a mathematical representation that a model understands.

model_name = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
sentiment = classifier("I Love HuggingFace")
print(sentiment)

sequence = "Buju Banton is the GOAT."
result = tokenizer(sequence)
print(result)

tokens = tokenizer.tokenize(sequence)
print(tokens)

ids = tokenizer.convert_tokens_to_ids(tokens)
print(ids)

decoded_string = tokenizer.decode(ids)
print(decoded_string)