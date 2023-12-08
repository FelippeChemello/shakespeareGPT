import torch
import torch.nn as nn
from torch.nn import functional as F

percent_of_training = 0.9
context_window = 8
batch_size = 4
learning_rate = 1e-3
evaluation_iterations = 200
evaluate_interval = 250
training_iterations = 5000
number_of_embeddings = 32
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

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(evaluation_iterations)
        for k in range(evaluation_iterations):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(number_of_embeddings, head_size, bias=False)
        self.query = nn.Linear(number_of_embeddings, head_size, bias=False)
        self.value = nn.Linear(number_of_embeddings, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(context_window, context_window)))

    def forward(self, x):
        batch_size, context_window_size, number_of_embeddings = x.shape
        keys = self.key(x)
        queries = self.query(x)

        # Compute the attention scores, affinities between the keys and the queries
        wei = queries @ keys.transpose(-2, -1) * number_of_embeddings**-0.5 
        # Results in (batch_size, context_window_size, context_window_size)
        # That means for each token in the context window, we have a score for each other token in the context window
        
        # Apply the mask to the attention scores
        wei = wei.masked_fill(self.tril[:context_window_size, :context_window_size] == 0, float('-inf'))
        # We replace the zeros with -inf, so that when we apply the softmax, the score becomes zero
        
        # Apply the softmax to the attention scores
        wei = F.softmax(wei, dim=-1)

        # Agrregate the values, by multiplying the scores with the values
        values = self.value(x)
        out = wei @ values

        return out
    
class MultiHeadAttention(nn.Module):
    def __init__(self, number_of_heads, head_size):
        super().__init__()
        # Create the heads
        self.heads = nn.ModuleList([Head(head_size) for _ in range(number_of_heads)])

        self.projection = nn.Linear(number_of_embeddings, number_of_embeddings)

    def forward(self, x):
        output = torch.cat([head(x) for head in self.heads], dim=-1)
        output = self.projection(output)
        return output

class FeedForward(nn.Module):
    def __init__(self, number_of_embeddings):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(number_of_embeddings, 4 * number_of_embeddings), # Multiplies by 4 to make the inner dimension bigger
            nn.ReLU(),
            nn.Linear(4 * number_of_embeddings, number_of_embeddings) # Reduces the dimension back to the original
        )

    def forward(self, x):
        return self.net(x)
    
class Block(nn.Module):
    def __init__(self, number_of_embeddings, number_of_heads):
        super().__init__()
        head_size = number_of_embeddings // number_of_heads
        # Get the connection between the tokens in the context window
        self.self_attention = MultiHeadAttention(number_of_heads, head_size)

        # Computes the attention scores between the tokens in the context window
        self.feed_forward = FeedForward(number_of_embeddings)

    def forward(self, x):
        # We sum the input with the output of the self attention and the feed forward
        # To represent the residual connection

        x = x + self.self_attention(x)
        x = x + self.feed_forward(x)
        return x

class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Changed from vocab size to number of embeddings, to make it more efficient
        self.token_embedding_table = nn.Embedding(vocab_size, number_of_embeddings) 
        
        # Create a linear layer to predict the next token from the embeddings
        self.language_model_head = nn.Linear(number_of_embeddings, vocab_size)

        # Create a positional embedding table, to make the model aware of the position of the tokens
        self.positional_embedding_table = nn.Embedding(context_window, number_of_embeddings)

        self.blocks = nn.Sequential(
            Block(number_of_embeddings, number_of_heads=4),
            Block(number_of_embeddings, number_of_heads=4),
            Block(number_of_embeddings, number_of_heads=4),
        )

    def forward(self, idx, targets=None):
        token_embeddings = self.token_embedding_table(idx)

        # Array of shape (context_window, number_of_embeddings)
        position_indexes = torch.arange(idx.shape[1]).to(device)
        position_embeddings = self.positional_embedding_table(position_indexes) 

        x = token_embeddings + position_embeddings # Add the two embeddings together - (batch_size, context_window, number_of_embeddings) 
        # X represents now the embeddings of the tokens and their position in the context window

        x = self.blocks(x)

        logits = self.language_model_head(x)

        if targets is None:
            return logits, None
        
        batch_size, context_window_size, vocab_size = logits.shape
        logits = logits.view(batch_size * context_window_size, vocab_size)
        targets = targets.view(batch_size * context_window_size)

        loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_on_context_window = idx[:, -context_window:]

            logits, _ = self.forward(idx_on_context_window)
            logits = logits[:, -1, :]

            probs = F.softmax(logits, dim=-1)

            idx_new_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_new_token], dim=1)
        return idx
    
model = BigramLanguageModel().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for iteration in range(training_iterations):
    if iteration % evaluate_interval == 0:
        losses = estimate_loss()
        print(f'Iteration: {iteration}, Train loss: {losses["train"]}, Val loss: {losses["val"]}')
    
    x, y = get_batch('train')

    logits, loss = model(x, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

initial_idx = torch.zeros((1, 1), dtype=torch.long).to(device)
post_training_idx = model.generate(initial_idx, max_new_tokens=500)
print(f"Text generated after training: {decode(post_training_idx[0].tolist())}")
