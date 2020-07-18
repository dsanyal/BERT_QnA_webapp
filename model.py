import tensorflow as tf
from transformers import BertTokenizer, TFBertForQuestionAnswering

print("#### Downloading BERT tokenizer ####")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokenizer.save_pretrained("models/tokenizer")
print("#### Downloading TFBertForQuestionAnswering model ####")
model = TFBertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
model.save_pretrained("models/")

print('#### Done ####')