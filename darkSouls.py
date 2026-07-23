import torch # we use PyTorch: https://pytorch.org
import torch.nn as nn
from torch.nn import functional as F
import gradio as gr


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
    
#tokenizer functions
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])




data = torch.tensor(encode(text), dtype=torch.long)
split = int(len(data)*0.9)
data_train = data[:split]
data_test = data[split:]




def get_batch(split):
    """
    generate a small random batch of data of inputs x and targets y
    """
    data = data_train if split == 'train' else data_test
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix]) #stacks tensors of the same shape along a new dimension
    #in this case the tensors are from our data (tensor of the text)
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]) #shifted right by 1 (we're trying to predict the NEXT token)
    x, y = x.to(device), y.to(device) #moves from cpu memory to device/intended memory
    return x, y


xb, yb = get_batch('train')
#print(xb.shape)  # should be (batch_size, block_size)
#print(yb.shape)  # should be (batch_size, block_size)

@torch.no_grad() #means we wont calculate gradients to save memory
def estimate_loss():
    """averages loss for train and val"""
    out = {}
    m.eval() #evaluation mode (turns off training dropout)
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters) #storing all losses 
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = m(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean() #gets the mean/average
    m.train() #back to training mode
    return out



cross_entropy = nn.CrossEntropyLoss()

class Head(nn.Module):
    """head of self-attention"""
    
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False) # key: what kind of info the token contains (makes it searchable)
        self.query = nn.Linear(n_embd, head_size, bias=False) #query: what is the token looking for from the rest of the text
        self.value = nn.Linear(n_embd, head_size, bias=False) #value: the actual content, passed along if the key matches a query
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size))) #persistent buffer
        # register_buffer essentially is part of the model but is not learned
        # in this case 'tril' is the lower-triangular matrix mask used to prevent tokens from seeing the future
        self.dropout = nn.Dropout(dropout) # helps stop overfitting
        #sets some neurons to 0 in each training step, meaning the model ant rely on any single neuron so it wont just memorize the training set

    def forward(self, x): #called automatically when you do head(x) or something similar
        """
        Tokens produce query and key. We will compute their scores with q @ k.T (comparing all queries againt all keys
        Softmax makes scores into probabilities.
        Tokens with high key-query similarity (high weight) contribute more to the output bc we take a weighted sum of the values 
        """
       
        B,T,C = x.shape
        k = self.key(x) # (B, T, C)
        q = self.query(x) # (B, T, C) 
        # compute attention scores ("affinities")
        # normalizing it
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B, T, C) @ (B, C, T) = (B, T, T)
        # scaled by C**-0.5 (sqrt of head_size) to prevent large dot products
        # and to keep values at reasonable levels so softmax isn't pushed into flat regions that would have gradients near 0
        
        # decoder block
        wei = wei.masked_fill(self.tril[:T, :T] ==0, float('-inf')) # (B, T, T)
        #this is the causal mask, which forces tokens to only attend to themselves or previous tokens, never future ones
        
        # softmax
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        # perform the weighted aggregation of the values
        wei = self.dropout(wei) # randomly prevent some nodes from commuinicating by setting them to 0
        
        v = self.value(x) # (B, T, C)
        out = wei @ v # (B, T, T) @ (B, T, C) = (B, T, C)
        # weighted sum of the values
        # basically token's value vector is multiplied by its attention score and then the products are added together to make the output
        return out
        
        
class MultiHeadAttention(nn.Module):
    """
    Communication: heads of self-attention in parallel
    Tokens talk to each other through heads and gather info from other tokens
    """       
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList((Head(head_size) for _ in range(num_heads))) #list of heads
        #heads are registered as submodules so they show up in m.parameters and are considered part of the model
        self.proj = nn.Linear(n_embd, n_embd)  # projects concatenated head outputs back into residual pathway
        self.dropout = nn.Dropout(dropout) #regularization, randomly zeroes out some elements so that we dont focus too hard on specific neurons
        # regularization is a technique to prevent overfitting by adding a penalty term to the loss function :)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) # concatenates all outputs over channel dimension
        out = self.dropout(self.proj(out))  #linear transformation and regularization
        return out #output of communication

class FeedForward(nn.Module):
    """
        Each token independently processes what it gathered from attention (computation)
        4 layers: Linear -> ReLU -> Linear -> Dropout
        The hidden layer expands to 4x n_embd before projecting back. Higher space gives model more room for computation
        """
    
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
        #rescales each token vector to have mean of 0 and variance of 1

    def forward(self, x): # done by tokens independently 
        x = x + self.sa(self.ln1(x)) # fork off, do communication and come back
        x = x + self.ffwd(self.ln2(x)) # fork off, do computation and come back
        return x

class GPT(nn.Module):
    """
    Character-level GPT model.
    Token and position embeddings are added and passed through n_layer transformer blocks.
    Each block does: attention (tokens communicate) then feedforward (tokens compute independently)
    Final LayerNorm + linear head converts to logits over the vocabulary to get vocab scores.
    """

    def __init__(self, vocab_size):
        super().__init__() # runs nn.Module's init method to set up data structures
        # tokens read of logits for next token from the lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd) #maps tokens to learned vectors (rep. what characters "mean") of size n_embd
        self.position_embedding_table = nn.Embedding(block_size, n_embd) # maps positions (0 to block_size-1) to learned vectors of size n_embd (gives model info on where each token is in the sequence)
        
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)]) # stacks n_layer transformer blocks in sequence
        #self.blocks(x) runs through all 6 automatically
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size) #language modeling head. COnverts to vocab scores
        #lm_head projects from n_embd to vocab_size (384->67). 
        
    def forward(self, idx, targets=None):
        #idx and target are tensor of integers

        B, T = idx.shape # B = batch size, T = sequence length (number of tokens in context)
        tok_emb = self.token_embedding_table(idx) # (B, T, C), rep. what each token is 
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device)) # (T, C) where each token is
        x = tok_emb + pos_emb # (B, T, C) combining identity + pos.

        x = self.blocks(x) # (B, T, C) attention + feedforward x n_layer
        x = self.ln_f(x) # final layernorm before proj. to vocab

        logits = self.lm_head(x)  # (B, T, vocab_size) scores for possible next token (vocab scores)
        
        if targets is None:
            loss = None # means we're in generation mode so no targets just return logits
        else:
            B, T, C = logits.shape # B*T is N, C is vocab size
            logits = logits.view(B*T, C)   # reshape to (N, vocab_size) for cross_entropy
            targets = targets.view(B*T)    # reshape to (N,) bc cross_entropy expects 1D targets
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
    
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        #idx is (B, T) array of indices in current context
        
        
        for i in range(max_new_tokens):
            idx_cond = idx[:, -block_size:] #cropped to not go out of range
            # get predictions
            logits, loss = self(idx_cond) # goes to forward function
            # focuses on last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # get probabilities
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float('-inf')
            probs = F.softmax(logits / temperature, dim=-1) #(B, C)
            # sample from distribution            
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

m = GPT(vocab_size).to(device)
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


print("Default Output: ")
print(decode(m.generate(idx, max_new_tokens=500, temperature=1.0, top_k=None)[0].tolist())) #generates 500 tokens after idx
print()
print("High Temperature Output: ")
print(decode(m.generate(idx, max_new_tokens=500, temperature=2.0, top_k=None)[0].tolist())) #generates 500 tokens after idx
print()
print("Low Temperature Output: ")
print(decode(m.generate(idx, max_new_tokens=500, temperature=0.5, top_k=None)[0].tolist())) #generates 500 tokens after idx
print()
print("Top K Output: ")
print(decode(m.generate(idx, max_new_tokens=500, temperature=1.0, top_k=20)[0].tolist())) #generates 500 tokens after idx


def generate_text(prompt, temperature, max_tokens):
    context = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
    return decode(m.generate(context, int(max_tokens), temperature=temperature)[0].tolist())

gr.Interface(
    fn=generate_text,
    inputs=[
        gr.Textbox(label="Prompt"),
        gr.Slider(0.1, 2.0, value=1.0, label="Temperature"),
        gr.Slider(50, 500, value=200, step=1, label="Max Tokens")
    ],
    outputs=gr.Textbox(label="Generated Text"),
    title="DarkSoulsGPT"
).launch()