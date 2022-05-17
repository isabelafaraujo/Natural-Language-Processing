from transformers import AutoTokenizer, AutoModelForMaskedLM

#portuguese model
tokenizer = AutoTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")
model = AutoModelForMaskedLM.from_pretrained("neuralmind/bert-base-portuguese-cased")

#example sentence
sequence = f"O governo de {tokenizer.mask_token} liberou o acesso às plataformas."

input = tokenizer.encode(sequence, return_tensors="pt")
mask_token_index = torch.where(input == tokenizer.mask_token_id)[1]

token_logits = model(input)[0]
mask_token_logits = token_logits[0, mask_token_index, :]

top_tokens = torch.topk(mask_token_logits, 10, dim=1).indices[0].tolist() #mudar a quantidade de sugestões

#complete sentences
for token in top_tokens:
    print(sequence.replace(tokenizer.mask_token, tokenizer.decode([token])))
    
#suggestion tokens
for token in top_tokens:
    print(tokenizer.decode([token]))
