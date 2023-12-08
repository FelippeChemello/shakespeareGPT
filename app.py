with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

char_to_int = {c: i for i, c in enumerate(chars)}
int_to_char = {i: c for i, c in enumerate(chars)}

encode = lambda string: [char_to_int[char] for char in string]
decode = lambda ints: ''.join([int_to_char[i] for i in ints])

print("Encoded: ", encode('Hello World'))
print("Decoded: ", decode(encode('Hello World')))