# model.py

import torch.nn as nn

class SimpleRNNModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super(SimpleRNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.RNN(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.rnn(x)
        out = self.fc(out)
        return out
