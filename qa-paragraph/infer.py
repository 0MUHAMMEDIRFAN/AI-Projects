from transformers import BertForQuestionAnswering, BertTokenizerFast
import torch

# Load model and tokenizer
model = BertForQuestionAnswering.from_pretrained('./model')
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

# Context and question
context = "Mahatma Gandhi was born in 1869. He fought for India's independence using non-violence."
question = "What method did Gandhi use?"

# Tokenize
inputs = tokenizer(question, context, return_tensors='pt')
outputs = model(**inputs)

start = torch.argmax(outputs.start_logits)
end = torch.argmax(outputs.end_logits) + 1

answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][start:end]))
print("Answer:", answer)
