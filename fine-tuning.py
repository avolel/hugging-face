from transformers import Trainer, TrainingArguments

# 1. Prepare dataset
# 2. Load pretrained Tokenizer, call it with dataset and get the encoding
# 3. Build Pytorch dataset with encodings
# 4. Load pretrained model
# 5. a) Load trainer and train int
#    b) native pytorch training loop

training_args = TrainingArguments("test-trainer")
trainer = Trainer(model, 
                  training_args, 
                  train_dataset=tokenized_datasets["train"], 
                  eval_dataset=tokenized_datasets["validation"], 
                  data_collator=data_collector, 
                  tokenizer=tokenizer)

trainer.train()