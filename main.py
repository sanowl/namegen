# main.py

from data_generation.py import generate_synthetic_names
from dataset import NameDataset
from model import SimpleRNNModel
from train import train_model

def main():
    # Generate synthetic names
    synthetic_names = generate_synthetic_names(1000)

    # Create dataset
    dataset = NameDataset(synthetic_names)

    # Initialize model
    vocab_size = len(set(''.join(synthetic_names)))
    embed_size = 64
    hidden_size = 128
    model = SimpleRNNModel(vocab_size, embed_size, hidden_size)

    # Train the model
    train_model(model, dataset, num_epochs=20)

if __name__ == "__main__":
    main()
