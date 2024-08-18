# main.py

from data import generate_synthetic_names
from dataset import NameDataset
from lanaguge import SimpleRNNModel
from train import train_model

def main():
    # Generate synthetic names
    synthetic_names = generate_synthetic_names(1000)

    # Create dataset with padding token
    dataset = NameDataset(synthetic_names, pad_token='<PAD>')

    # Initialize model
    vocab_size = len(dataset.char_to_idx)
    embed_size = 64
    hidden_size = 128
    model = SimpleRNNModel(vocab_size, embed_size, hidden_size)

    # Train the model
    train_model(model, dataset, num_epochs=20)

if __name__ == "__main__":
    main()
