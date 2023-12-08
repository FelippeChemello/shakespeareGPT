import torch

percent_of_training = 0.9
context_window = 8
batch_size = 4
device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(777)

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

char_to_int = {c: i for i, c in enumerate(chars)}
int_to_char = {i: c for i, c in enumerate(chars)}

encode = lambda string: [char_to_int[char] for char in string]
decode = lambda ints: ''.join([int_to_char[i] for i in ints])

data = torch.tensor(encode(text), dtype=torch.long)
n_chars_for_training = int(len(data) * percent_of_training)
train_data = data[:int(n_chars_for_training)]
test_data = data[int(n_chars_for_training):]

def get_batch(split): 
    data = train_data if split == 'train' else test_data
    split_indexes = torch.randint(len(data) - context_window, (batch_size,))
    x = torch.stack([data[index:index+context_window] for index in split_indexes])
    y = torch.stack([data[index+1:index+context_window+1] for index in split_indexes])
    return x.to(device), y.to(device)

x, y = get_batch('train')
for batch in range(batch_size):
    for time in range(context_window):
        context = x[batch, :time+1]
        target = y[batch, time]
        print(f'Context: {context.tolist()} -> Target: {target}')
