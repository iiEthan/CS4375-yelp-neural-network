import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import random
import os
import time
from tqdm import tqdm
import json
from argparse import ArgumentParser
import matplotlib.pyplot as plt

unk = '<UNK>'

class FFNN(nn.Module):
    def __init__(self, input_dim, h):
        super(FFNN, self).__init__()
        self.h = h
        # First fully connected layer from input to hidden layer
        self.W1 = nn.Linear(input_dim, h)
        # ReLU activation function for introducing non-linearity
        self.activation = nn.ReLU()
        # Second fully connected layer from hidden to output layer (5 classes)
        self.output_dim = 5
        self.W2 = nn.Linear(h, self.output_dim)
        # LogSoftmax for final layer to provide log probabilities for each class
        self.softmax = nn.LogSoftmax(dim=0)
        # Loss function: Negative Log-Likelihood Loss for classification tasks
        self.loss = nn.NLLLoss()

    # Compute the loss given the predicted and actual labels
    def compute_Loss(self, predicted_vector, gold_label):
        return self.loss(predicted_vector, gold_label)

    # Forward pass through the network
    def forward(self, input_vector):
        # Apply the first layer and activation function
        h = self.activation(self.W1(input_vector))
        # Pass the result through the second layer
        z = self.W2(h)
        # Apply softmax to get class probabilities
        predicted_vector = self.softmax(z)
        return predicted_vector

# Create vocabulary from dataset
def make_vocab(data):
    vocab = set()
    for document, _ in data:
        for word in document:
            vocab.add(word)
    return vocab

# Convert vocabulary to indices for encoding
def make_indices(vocab):
    vocab_list = sorted(vocab)
    vocab_list.append(unk)
    word2index = {}
    index2word = {}
    for index, word in enumerate(vocab_list):
        word2index[word] = index
        index2word[index] = word
    vocab.add(unk)
    return vocab, word2index, index2word

# Convert dataset to vectorized format based on vocabulary indices
def convert_to_vector_representation(data, word2index):
    vectorized_data = []
    for document, y in data:
        vector = torch.zeros(len(word2index))
        for word in document:
            index = word2index.get(word, word2index[unk])
            vector[index] += 1
        vectorized_data.append((vector, y))
    return vectorized_data

# Load and process training and validation data
def load_data(train_data, val_data):
    with open(train_data) as training_f:
        training = json.load(training_f)
    with open(val_data) as valid_f:
        validation = json.load(valid_f)

    tra = []
    val = []
    for elt in training:
        tra.append((elt["text"].split(), int(elt["stars"] - 1)))
    for elt in validation:
        val.append((elt["text"].split(), int(elt["stars"] - 1)))

    return tra, val

# Main script execution for training and evaluating FFNN
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-hd", "--hidden_dim", type=int, required=True, help="hidden_dim")
    parser.add_argument("-e", "--epochs", type=int, required=True, help="num of epochs to train")
    parser.add_argument("--train_data", required=True, help="path to training data")
    parser.add_argument("--val_data", required=True, help="path to validation data")
    parser.add_argument("--test_data", default="to fill", help="path to test data")
    parser.add_argument('--do_train', action='store_true')
    args = parser.parse_args()

    random.seed(42)
    torch.manual_seed(42)

    # Load and preprocess data
    print("========== Loading data ==========")
    train_data, valid_data = load_data(args.train_data, args.val_data)
    vocab = make_vocab(train_data)
    vocab, word2index, index2word = make_indices(vocab)

    # Convert text data to vectors
    print("========== Vectorizing data ==========")
    train_data = convert_to_vector_representation(train_data, word2index)
    valid_data = convert_to_vector_representation(valid_data, word2index)

    # Initialize model and optimizer
    model = FFNN(input_dim=len(vocab), h=args.hidden_dim)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    training_losses = []
    validation_accuracies = []

    print("========== Training for {} epochs ==========".format(args.epochs))
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        epoch_loss = 0
        correct = 0
        total = 0
        start_time = time.time()
        random.shuffle(train_data)
        minibatch_size = 16
        N = len(train_data)
        # Mini-batch training loop
        for minibatch_index in tqdm(range(N // minibatch_size), desc=f"Training Epoch {epoch + 1}"):
            optimizer.zero_grad()
            loss = None
            for example_index in range(minibatch_size):
                input_vector, gold_label = train_data[minibatch_index * minibatch_size + example_index]
                predicted_vector = model(input_vector)
                predicted_label = torch.argmax(predicted_vector)
                correct += int(predicted_label == gold_label)
                total += 1
                example_loss = model.compute_Loss(predicted_vector.view(1, -1), torch.tensor([gold_label]))
                if loss is None:
                    loss = example_loss
                else:
                    loss += example_loss
            loss = loss / minibatch_size
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
        training_losses.append(epoch_loss / (N // minibatch_size))
        train_accuracy = correct / total
        print(f"Epoch {epoch + 1} Training Accuracy: {train_accuracy:.4f}, Loss: {epoch_loss / (N // minibatch_size):.4f}")

        # Validation loop
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for minibatch_index in tqdm(range(len(valid_data) // minibatch_size), desc="Validating"):
                for example_index in range(minibatch_size):
                    input_vector, gold_label = valid_data[minibatch_index * minibatch_size + example_index]
                    predicted_vector = model(input_vector)
                    predicted_label = torch.argmax(predicted_vector)
                    correct += int(predicted_label == gold_label)
                    total += 1
        validation_accuracy = correct / total
        validation_accuracies.append(validation_accuracy)
        print(f"Epoch {epoch + 1} Validation Accuracy: {validation_accuracy:.4f}")

    # Save final model performance and learning curve
    os.makedirs('results', exist_ok=True)
    with open('results/test.out', 'w') as f:
        f.write("Final Validation Accuracy: {}\n".format(validation_accuracies[-1]))

    # Plot learning curves
    plt.plot(range(1, args.epochs + 1), training_losses, label="Training Loss")
    plt.plot(range(1, args.epochs + 1), validation_accuracies, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Learning Curve")
    plt.legend()
    plt.savefig("results/learning_curve_ffnn.png")
