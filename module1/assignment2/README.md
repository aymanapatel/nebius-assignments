# Module 1: Assignment 2 - RNN Language Models & Image Classification

This repository contains the solution for Module 1 Assignment 2, covering Recurrent Neural Networks (RNN) for Language Modeling and Multi-layer Perceptrons (MLP) for CIFAR-10 image classification.

## Project Structure

- [RNN_LM_homework.ipynb](RNN_LM_homework.ipynb): Implementation of a character-level RNN language model.
- [homework_CIFAR10.ipynb](homework_CIFAR10.ipynb): Implementation of image classification tasks using the CIFAR-10 dataset.
- [Insights_Classification.md](Insights_Classification.md): Detailed analysis and performance metrics for the image classification experiments.
- [dinos.txt](dinos.txt): Dataset containing dinosaur names used for training the RNN language model.
- [AGENTS.md](AGENTS.md): Development notes and instructions for AI-assisted experimentation.

## Key Tasks

### 1. RNN Language Modeling
- Built a character-level RNN to generate text (e.g., dinosaur names).
- Implemented the forward pass, loss computation, and sampling logic.
- Trained the model on [dinos.txt](dinos.txt) to learn name structures and patterns.

### 2. CIFAR-10 Image Classification
- **Binary Classification**:
  - Airplane vs. Ship: Achieved **94.35%** test accuracy (Target: >94%).
  - Cat vs. Dog: Achieved **65.35%** test accuracy (Target: >64%).
- **Multi-class Classification**:
  - Classified all 10 CIFAR-10 categories.
  - Achieved **56.22%** test accuracy (Target: >53%).
- **Experiments**:
  - Compared different activation functions (`sigmoid`, `tanh`, `relu`).
  - Analyzed the impact of network depth, Batch Normalization, and Dropout.
  - Implemented custom `CIFAR10Dataset` for efficient data handling.

## Performance Summary

| Task | Test Accuracy | Target | Status |
|---|---|---|---|
| Airplane vs Ship | 94.35% | >94% | ✅ Passed |
| Cat vs Dog | 65.35% | >64% | ✅ Passed |
| CIFAR-10 (10 classes) | 56.22% | >53% | ✅ Passed |

For a more in-depth breakdown of activation comparisons and architectural findings, see [Insights_Classification.md](Insights_Classification.md).

## Usage

1. Open the `.ipynb` files in a Jupyter environment.
2. Ensure `torch`, `torchvision`, and `matplotlib` are installed.
3. Run the cells sequentially to reproduce the training and evaluation results.
