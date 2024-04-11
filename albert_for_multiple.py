from transformers import AlbertTokenizer, BertForMultipleChoice

tokenizer = AlbertTokenizer.from_pretrained("albert-base-v2")

model = BertForMultipleChoice.from_pretrained('albert-base-v2', return_dict=True)

sentence = 'I love this movie!'
question = 'Do you like this movie?'
options = ['It is the best movie ever.', 'It is just so-so.', 'I hate it.']  # Options for each class

# encoded_inputs = [tokenizer(sentence, option, padding=True, truncation=True, return_tensors='pt') for option in options]
encoded_inputs = tokenizer(sentence, question, options, padding=True, truncation=True, return_tensors='pt')

outputs = model(**encoded_inputs)
logits = outputs.logits
print(logits)  # tensor([[-0.3971,  0.2948,  0.5706]], grad_fn=<AddmmBackward>)
