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
import string
from argparse import ArgumentParser
import pickle
import matplotlib.pyplot as plt

unk = '<UNK>'

class RNN(nn.Module):
    def __init__(self, input_dim, h):
        super(RNN, self).__init__()
        self.h = h
        self.numOfLayer = 1
        # Define RNN layer with tanh activation
        self.rnn = nn.RNN(input_dim, h, self.numOfLayer, nonlinearity='tanh')
        # Define dropout layer to prevent overfitting
        self.dropout = nn.Dropout(0.5)
        # Linear layer to map RNN outputs to 5 output classes
        self.W = nn.Linear(h, 5)
        # LogSoftmax for output layer to provide log probabilities
        self.softmax = nn.LogSoftmax(dim=1)
        # Loss function: Negative Log-Likelihood Loss for classification tasks
        self.loss = nn.NLLLoss()

    # Compute the loss between predicted and actual labels
    def compute_Loss(self, predicted_vector, gold_label):
        return self.loss(predicted_vector, gold_label)

    # Forward pass through the network
    def forward(self, inputs):
        # Get hidden layer output from RNN
        _, hidden = self.rnn(inputs)
        # Apply dropout and then map hidden state to output layer
        z = self.W(self.dropout(hidden[-1]))
        # Apply softmax to get class probabilities
        predicted_vector = self.softmax(z)
        return predicted_vector


# Load and preprocess training and validation data
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


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-hd", "--hidden_dim", type=int, required=True, help="hidden_dim")
    parser.add_argument("-e", "--epochs", type=int, required=True, help="num of epochs to train")
    parser.add_argument("--train_data", required=True, help="path to training data")
    parser.add_argument("--val_data", required=True, help="path to validation data")
    parser.add_argument("--test_data", default="to fill", help="path to test data")
    parser.add_argument('--do_train', action='store_true')
    args = parser.parse_args()

    print("========== Loading data ==========")
    train_data, valid_data = load_data(args.train_data, args.val_data)

    print("========== Vectorizing data ==========")
    # Initialize RNN model and optimizer
    model = RNN(50, args.hidden_dim)
    optimizer = optim.SGD(model.parameters(), lr=0.0005, momentum=0.9)

    # Define a learning rate scheduler to adjust learning rate every 5 epochs
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # Load precomputed word embeddings
    word_embedding = pickle.load(open('./word_embedding.pkl', 'rb'))

    training_losses = []
    validation_accuracies = []

    stopping_condition = False
    epoch = 0
    last_train_accuracy = 0
    last_validation_accuracy = 0

    # Training loop
    while epoch < args.epochs and not stopping_condition:
        random.shuffle(train_data)
        model.train()
        print("Training started for epoch {}".format(epoch + 1))
        correct = 0
        total = 0
        minibatch_size = 42
        N = len(train_data)
        epoch_loss = 0

        # Mini-batch training loop
        for minibatch_index in tqdm(range(N // minibatch_size), desc=f"Training Epoch {epoch + 1}"):
            optimizer.zero_grad()
            loss = None
            for example_index in range(minibatch_size):
                input_words, gold_label = train_data[minibatch_index * minibatch_size + example_index]
                input_words = " ".join(input_words)
                input_words = input_words.translate(input_words.maketrans("", "", string.punctuation)).split()

                # Convert words to vectors using word embeddings
                vectors = [word_embedding[i.lower()] if i.lower() in word_embedding else word_embedding['unk'] for i in
                           input_words]
                vectors = torch.tensor(np.array(vectors)).view(len(vectors), 1, -1)

                # Forward pass through the RNN
                output = model(vectors)

                # Compute loss and track accuracy
                example_loss = model.compute_Loss(output.view(1, -1), torch.tensor([gold_label]))
                predicted_label = torch.argmax(output)
                correct += int(predicted_label == gold_label)
                total += 1
                if loss is None:
                    loss = example_loss
                else:
                    loss += example_loss

            # Average the loss over the mini-batch and perform backpropagation
            loss = loss / minibatch_size
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()

        # Record training loss for learning curve
        training_losses.append(epoch_loss / (N // minibatch_size))
        train_accuracy = correct / total
        print(
            f"Epoch {epoch + 1} Training Accuracy: {train_accuracy:.4f}, Loss: {epoch_loss / (N // minibatch_size):.4f}")

        # Validation loop to evaluate model on validation data
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for input_words, gold_label in tqdm(valid_data, desc="Validating"):
                input_words = " ".join(input_words)
                input_words = input_words.translate(input_words.maketrans("", "", string.punctuation)).split()
                vectors = [word_embedding[i.lower()] if i.lower() in word_embedding else word_embedding['unk'] for i in
                           input_words]
                vectors = torch.tensor(np.array(vectors)).view(len(vectors), 1, -1)
                output = model(vectors)
                predicted_label = torch.argmax(output)
                correct += int(predicted_label == gold_label)
                total += 1
        validation_accuracy = correct / total
        validation_accuracies.append(validation_accuracy)
        print(f"Epoch {epoch + 1} Validation Accuracy: {validation_accuracy:.4f}")

        # Step the learning rate scheduler to reduce learning rate if needed
        scheduler.step()

        # Early stopping condition based on validation accuracy
        if validation_accuracy < last_validation_accuracy and train_accuracy > last_train_accuracy:
            print("Training done to avoid overfitting!")
            print("Best validation accuracy is:", last_validation_accuracy)
        else:
            last_validation_accuracy = validation_accuracy
            last_train_accuracy = train_accuracy

        epoch += 1

    # Save final results and plot the learning curve
    os.makedirs('results', exist_ok=True)
    with open('results/rnn_test.out', 'w') as f:
        f.write("Final Validation Accuracy: {}\n".format(validation_accuracies[-1]))

    # Plot training loss and validation accuracy over epochs
    plt.plot(range(1, epoch + 1), training_losses, label="Training Loss")
    plt.plot(range(1, epoch + 1), validation_accuracies, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Learning Curve")
    plt.legend()
    plt.savefig("results/learning_curve_rnn.png")
