# train.py

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch import nn
from lanaguge import SimpleRNNModel
from dataset import NameDataset

def train_model(model, dataset, num_epochs=20, batch_size=32, learning_rate=0.001):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss(reduction='none')  # We'll handle reduction manually
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        for inputs, targets, mask in dataloader:
            outputs = model(inputs)
            outputs = outputs.view(-1, model.fc.out_features)
            targets = targets.view(-1)
            mask = mask.view(-1)

            # Calculate loss for each element
            loss = criterion(outputs, targets)

            # Apply the mask to ignore padding
            loss = loss * mask

            # Sum the loss over all non-padded elements and normalize
            loss = loss.sum() / mask.sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    print("Training finished.")
