from transformers import pipeline

classifier = pipeline("zero-shot-classification")

zero_shot_classi = classifier("I love chicken and rice.", candidate_labels=["food","travel","business"])

print(zero_shot_classi)