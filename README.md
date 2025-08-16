# Algoverse-AI-PTS-Model-Linear-Probing
## Linear Probing for Pivotal Token Prediction in Qwen Models

This repository contains experimental code to train linear probes that predict whether the next token is a "pivotal token" (as defined by the original dataset structure) based on the activations of the Qwen3-0.6B language model and datasets formed through **Pivotal Token Search**.

This project is an experiment in interpretability, aiming to understand if and where in the model's layers information about upcoming high impact tokens is linearly decipherable from the hidden states.

## Goal
Train a simple linear classifier for each layer of Qwen3-0.6B to predict the likelihood that the token immediately following the probed position is a pivotal token.

## Key Features
- Loading the Qwen3-0.6B model and tokenizer.
- Preparing and balancing a dataset of pivotal and non-pivotal token positions.
- Extracting layer activations at specified token positions using the full text context.
- Training a Linear Probe (linear layer + sigmoid) for multiple model layers.
- Saving the trained state dictionary for the best probe of each layer.
- Identifying and specially marking the overall best-performing layer probe based on test accuracy.
- Code examples for loading saved probes and performing inference on new text.
- Setup and Running
This code is designed to be run in a Google Colab notebook or a similar environment with GPU access.

## Experimentation and Analysis
This is an experimental project. The code provides the tools to train and save probes. Interesting next steps include:

- Analyzing Accuracy: Examine the layer_wise_accuracies to see which layers are best at predicting pivotal tokens.
- Loading Probes: Use the provided loading code to load probes from interesting layers.
- Inference: Use the loaded probes with the sample text prediction code to see how they predict on new, unseen text.
- Visualization: Extract activations for a subset of test data, load the best probe, score the activations with the probe, and visualize the activations (e.g., using PCA or t-SNE) colored by true label and/or probe score to see if the probe creates linearly separable clusters.
- Hyperparameter Tuning: Experiment with learning_rate, num_epochs, or batch_size in the train_and_evaluate_probe function.
Feel free to explore and modify the code to conduct further experiments!

## Model Reference
This project uses the **Qwen3-0.6B** model from Hugging Face.
