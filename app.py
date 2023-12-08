import torch
import torch.nn as nn
from torch.nn import functional as F

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

class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Each token directly reads the logits of the next token from an embedding table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # idx and targets are of shape (batch_size, context_window)
        
        # The output of the embedding table is of shape (batch_size, context_window, vocab_size)
        logits = self.token_embedding_table(idx)
        # Logits are the raw values produced by the last linear layer of a neural network.
        # Before activation function is applied.

        if targets is None:
            return logits, None
        
        # To calculate the loss we need to reshape the logits and targets to (batch_size * context_window, vocab_size)
        # https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html 
        batch_size, context_window_size, vocab_size = logits.shape
        logits = logits.view(batch_size * context_window_size, vocab_size)
        targets = targets.view(batch_size * context_window_size)

        loss = F.cross_entropy(logits, targets)

        return logits, loss
    
model = BigramLanguageModel().to(device)
x, y = get_batch('train')
logits, loss = model(x, y)
print("Logits shape:", logits.shape)
print("The loss obtained is {:.4f} which is proximate to -log(1/len(vocab)) = {:.4f}".format(loss, -torch.log(torch.tensor(1/vocab_size))))