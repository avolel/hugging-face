from transformers import pipeline

generator = pipeline("text-generation", model="distilgpt2")

gen_text = generator("Love is Love", max_length=30, num_return_sequences=2)

print(gen_text)