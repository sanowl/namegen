# train.py

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch as nn

def train_model(model, dataset, num_epochs=20, batch_size=32, learning_rate=0.001):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        for inputs, targets in dataloader:
            outputs = model(inputs)
            outputs = outputs.view(-1, model.fc.out_features)
            targets = targets.view(-1)

            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    print("Training finished.")
