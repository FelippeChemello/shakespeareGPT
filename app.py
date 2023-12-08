with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

print(text[:1000])
print("O tamanho do texto é de {} caracteres".format(len(text)))

chars = sorted(list(set(text)))
vocab_size = len(chars)
print("O texto possui {} caracteres únicos, eles são: {}".format(vocab_size, chars))