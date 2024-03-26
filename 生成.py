import numpy as np


#ステップ1 データ取得

with open('文章.txt', 'r', encoding='utf8') as fp:
    text = fp.read()

start_index = text.find('THE MYSTERIOUS ISLAND')
end_index = text.find('End of the Project Gutenberg')
text = text[start_index:end_index]

#ステップ2 データの数値化
char_set = set(text) # 重複を削除
chars_sorted = sorted(char_set)
char2int = {ch:i for i, ch in enumerate(chars_sorted)}
char_array = np.array(chars_sorted)
text_encoded = np.array([char2int[ch] for ch in text], dtype=np.int32)
seq_length = 40
chunk_size = seq_length+1
text_chunks = [text_encoded[i:i+chunk_size] for i in range(len(text_encoded)-chunk_size+1)]


import torch
from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, text_chunks):
        self.text_chunks = text_chunks

    def __len__(self):
        return len(self.text_chunks)
    
    def __getitem__(self, idx):
        text_chunk = self.text_chunks[idx]
        return text_chunk[:-1].long(), text_chunk[1:].long()
    
seq_dataset = TextDataset(torch.tensor(text_chunks))

from torch.utils.data import DataLoader

batch_size = 64
seq_dl = DataLoader(seq_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

import torch.nn as nn

class Model(nn.Module):
    def __init__(self, vocab_size, embed_dim, rnn_hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn = nn.LSTM(embed_dim, rnn_hidden_size, batch_first=True)
        self.fc = nn.Linear(rnn_hidden_size, vocab_size)

    def forward(self, x, hidden, cell):
        out = self.embedding(x).unsqueeze(1)
        out, (hidden, cell) = self.rnn(out, (hidden, cell))
        out = self.fc(out).reshape(out.size(0), -1)
        return out, hidden, cell
    
    def init_hidden(self, batch_size):
        hidden = torch.zeros(1, batch_size, self.rnn_hidden_size)
        cell = torch.zeros(1, batch_size, self.rnn_hidden_size)
        return hidden, cell

vocab_size = len(char_array)
embed_dim = 256
rnn_hidden_size = 512
model = Model(vocab_size, embed_dim, rnn_hidden_size)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
num_epoch = 1000

for epoch in range(num_epoch):
    hidden, cell = model.init_hidden(batch_size)
    seq_batch, target_batch = next(iter(seq_dl))
    optimizer.zero_grad()
    loss = 0
    for c in range(seq_length):
        pred, hidden, cell = model(seq_batch[:, c], hidden, cell)
        loss += loss_fn(pred, target_batch[:, c])

    loss.backward()
    optimizer.step()
    loss = loss.item()/seq_length
    if epoch % 500==0:
        print(epoch, loss)

 