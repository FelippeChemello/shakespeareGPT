import torch
import torch.nn as nn
from torch.nn import functional as F

percent_of_training = 0.9
context_window = 8
batch_size = 4
learning_rate = 1e-2
evaluation_iterations = 200
evaluate_interval = 250
training_iterations = 1000
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

# Reduce the amount of memory used by the model
@torch.no_grad()
def estimate_loss():
    out = {}
    # We do not want to train the model while evaluating
    model.eval()

    for split in ['train', 'val']:
        losses = torch.zeros(evaluation_iterations)
        for k in range(evaluation_iterations):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()

    # We want to train the model after evaluating
    model.train()
    return out


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

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            # As we are generating a new token, we do not use the loss, as we do not have the targets
            logits, _ = self.forward(idx)

            # For this moment, we are implementing a bigram model, so we only care about the last token of the context window
            logits = logits[:, -1, :] # Shape (batch_size, vocab_size)

            probs = F.softmax(logits, dim=-1)

            idx_new_token = torch.multinomial(probs, num_samples=1) # Shape (batch_size, 1)
            idx = torch.cat([idx, idx_new_token], dim=1) # Shape (batch_size, context_window + 1)
        return idx
    
model = BigramLanguageModel().to(device)
initial_idx = torch.zeros((1, 1), dtype=torch.long).to(device)

before_training_idx = model.generate(initial_idx, max_new_tokens=100)
print(f"Text generated before training: {decode(before_training_idx[0].tolist())}")

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for iteration in range(training_iterations):
    if iteration % evaluate_interval == 0:
        losses = estimate_loss()
        print(f'Iteration: {iteration}, Train loss: {losses["train"]}, Val loss: {losses["val"]}')
    
    x, y = get_batch('train')

    logits, loss = model(x, y)

    # Clear the gradients of all optimized tensors
    optimizer.zero_grad()

    # Compute gradient of the loss with respect to model parameters
    loss.backward()

    # Perform a single optimization step
    optimizer.step()


post_training_idx = model.generate(initial_idx, max_new_tokens=100)
print(f"Text generated after training: {decode(post_training_idx[0].tolist())}")
