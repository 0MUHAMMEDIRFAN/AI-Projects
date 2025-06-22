from datasets import Dataset
from transformers import BertForQuestionAnswering, BertTokenizerFast, TrainingArguments, Trainer
import torch
import json

# Load and flatten your data
def load_squad_json(path):
    with open(path, "r") as f:
        data = json.load(f)

    examples = []
    for article in data["data"]:
        for paragraph in article["paragraphs"]:
            context = paragraph["context"]
            for qa in paragraph["qas"]:
                question = qa["question"]
                answer = qa["answers"][0]
                examples.append({
                    "context": context,
                    "question": question,
                    "answers": {"text": [answer["text"]], "answer_start": [answer["answer_start"]]}
                })
    return examples

examples = load_squad_json("data.json")
dataset = Dataset.from_list(examples)

# Load tokenizer
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

# Preprocessing
def preprocess(example):
    inputs = tokenizer(
        example["question"], example["context"],
        truncation="only_second", padding="max_length", max_length=512,
        return_offsets_mapping=True
    )

    # Compute start and end token indices
    offset = inputs.pop("offset_mapping")
    start_char = example["answers"]["answer_start"][0]
    end_char = start_char + len(example["answers"]["text"][0])

    sequence_ids = inputs.sequence_ids()
    context_start = sequence_ids.index(1)
    context_end = len(sequence_ids) - 1 - sequence_ids[::-1].index(1)

    start_token = end_token = 0
    for i in range(context_start, context_end):
        if offset[i][0] <= start_char and offset[i][1] >= start_char:
            start_token = i
        if offset[i][0] <= end_char and offset[i][1] >= end_char:
            end_token = i
            break

    inputs["start_positions"] = start_token
    inputs["end_positions"] = end_token
    return inputs

tokenized_dataset = dataset.map(preprocess, remove_columns=dataset.column_names)

# Model
model = BertForQuestionAnswering.from_pretrained("bert-base-uncased")

# Training setup
args = TrainingArguments(
    output_dir="./model",
    evaluation_strategy="no",
    per_device_train_batch_size=2,
    num_train_epochs=3,
    save_steps=10_000,
    save_total_limit=1
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_dataset
)

# Train
trainer.train()
trainer.save_model("./model")
