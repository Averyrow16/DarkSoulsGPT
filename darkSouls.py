import torch # we use PyTorch: https://pytorch.org
import torch.nn as nn
from torch.nn import functional as F
# hyperparameters:
batch_size = 64 # individual sequences being processed in parallel
block_size = 256 # maximum context length for predictions
max_iters = 1500
eval_interval = 300
learning_rate = 1e-3
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
device = 'cuda'

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU available")

with open('allDialogue.txt', 'r', encoding='utf-8') as file:
    text = file.read()

#unique characters
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = {}
itos = {}
for i, ch in enumerate(chars):
    stoi[ch] = i
    itos[i] = ch
    
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])




data = torch.tensor(encode(text), dtype=torch.long)
split = int(len(data)*0.9)
data_train = data[:split]
data_test = data[split:]



"""
generate a small random batch of data of inputs x and targets y
"""
def get_batch(split):
    data = data_train if split == 'train' else data_test
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


xb, yb = get_batch('train')
#print(xb.shape)  # should be (batch_size, block_size)
#print(yb.shape)  # should be (batch_size, block_size)

@torch.no_grad()
def estimate_loss():
    out = {}
    m.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = m(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    m.train()
    return out



cross_entropy = nn.CrossEntropyLoss()

class Head(nn.Module):
    """head of self-attention"""
    
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size))) #different from a parameter (persistent buffer)
        #register_buffer essentially is part of the model but is not learned
        # in this case 'tril' is the lower-triangular matrix mask used to prevent tokens from seeing the future
        self.dropout = nn.Dropout(dropout) # helps stop overfitting

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x) # (B, T, C)
        q = self.query(x) # (B, T, C) 
        # compute attention scores ("affinities")
        # normalizing it
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B, T, C) @ (B, C, T) = (B, T, T)
        # decoder block
        wei = wei.masked_fill(self.tril[:T, :T] ==0, float('-inf')) # (B, T, T)
        # softmax
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        # perform the weighted aggregation of the values
        wei = self.dropout(wei) # randomly prevent some nodes from commuinicating by setting them to 0
        
        v = self.value(x) # (B, T, C)
        out = wei @ v # (B, T, T) @ (B, T, C) = (B, T, C)
        return out
        
        
class MultiHeadAttention(nn.Module):
    """heads of self-attention in parallel"""       
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList((Head(head_size) for _ in range(num_heads))) #list of heads
        self.proj = nn.Linear(n_embd, n_embd) # linear transformation
        self.dropout = nn.Dropout(dropout) #regularization, randomly zeroes out some elements so that we dont focus too hard on specific neurons
    
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) # concatenates all outputs over channel dimension
        out = self.dropout(self.proj(out)) 
        return out

class FeedForward(nn.Module):
    """a simple linear layer followed by a non-linearity"""
    
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential( # chains layers so output of one feeds to next
            nn.Linear(n_embd, 4*n_embd), # dense layer that takes vector of size n_embd and outputs vector of size n_embd
            nn.ReLU(), # Rectified Linear Unit activation function (introduces non-linearity)
            nn.Linear(4*n_embd, n_embd), # projection layer going back into residual pathway
            nn.Dropout(dropout), # dropout is added right before residual connection connects back to original pathway
        )
        
    def forward(self, x): # all tokens do this independently
        return self.net(x)
# self attention is the communication, feed forward allows them to think on the communication individually

class Block(nn.Module):
    """ Transformer block (communication followed by computation)"""
    
    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size) # communication
        self.ffwd = FeedForward(n_embd) # computation
        self.ln1 = nn.LayerNorm(n_embd) #normalizes features
        self.ln2 = nn.LayerNorm(n_embd) # needs 2 because the learned parameters in each case are trying to fit different needs
    def forward(self, x): # done by tokens independently 
        x = x + self.sa(self.ln1(x)) # fork off, do communication and come back
        x = x + self.ffwd(self.ln2(x)) # fork off, do computation and come back
        return x

class BigramLanguageModel(nn.Module):
    
    def __init__(self, vocab_size):
        super().__init__() # runs nn.Module's init method to set up data structures
        # tokens read of logits for next token from the lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size) #language modeling head
        
    def forward(self, idx, targets=None):
        
        B, T = idx.shape
        #idx and target are tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb # (B, T, C)
        x = self.blocks(x) # (B, T, C)
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape # B*T is N, C is vocab size
            logits = logits.view(B*T, C)              # now (N, vocab_size), N = B*T
            targets = targets.view(B*T)               # now (N,)
            loss = F.cross_entropy(logits, targets)
            """
            counts = logits.exp()
            probs = counts / counts.sum(1, keepdims=True)  # now sum(1) is correct — sums over vocab
            loss2 = -probs[torch.arange(B*T), targets].log().mean()
            explicite version of loss so i can better understand it
            """
        #print(f"loss: {loss}")
        #print(f"loss2: {loss2}")
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        #idx is (B, T) array of indices in current context
        for i in range(max_new_tokens):
            idx_cond = idx[:, -block_size:] #cropped to not go out of range
            # get predictions
            logits, loss = self(idx_cond) # goes to forward function
            # focuses on last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from distribution            
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

m = BigramLanguageModel(vocab_size).to(device)
logits, loss = m(xb, yb)
#print(logits.shape)
#print(loss)
idx = torch.zeros((1, 1), dtype=torch.long, device=device) #zero starts off the generation (represents newline)
print(decode(m.generate(idx, max_new_tokens=100)[0].tolist())) #generates 100 tokens after idx

optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

batch_size = 32
best_val_loss  = 5.0
"""
for iter in range(max_iters):
    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    
        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']
            torch.save(m.state_dict(), 'best_model.pt') 
            
    # sample a batch of data
    xb, yb = get_batch('train')
    
    # evaluate the loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step() 
"""
m.load_state_dict(torch.load('best_model.pt'))
m.eval()

print(decode(m.generate(idx, max_new_tokens=500)[0].tolist())) #generates 100 tokens after idx
