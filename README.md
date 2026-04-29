# Nebius Academy: AI Performance Engineering - Assignments

This repository contains my submissions for the AI Performance Engineering course at Nebius Academy. It is organized by modules and weekly assignments covering LLM architectures, training techniques, and AI agents.

## Repository Structure

### [Module 1: LLM Architectures & Training](module1/)

Focuses on the fundamentals of Large Language Models, from basic MLP classification to building transformers from scratch.

- **[Assignment 1](module1/assignment1/)**: LLM Architectures study and basic implementation.
- **[Assignment 2](module1/assignment2/)**: RNN Language Models (character-level generation) and CIFAR-10 Image Classification (MLP optimization).
- **[Assignment 3](module1/assignment3/)**: Building a "Tiny Transformer" (GPT-style decoder) from scratch in PyTorch.

### [Module 2: AI Agents](module2-agents/)

Focuses on building and orchestrating AI agents using various frameworks.

- **[Edinburgh Agent (Week 1)](module2-agents/)**: 
  - Context management and prompts.
  - Graph-based agents with LangGraph.
  - Retrieval-augmented generation (RAG) concepts.
  - Dialog systems with Rasa.
  - MCP (Model Context Protocol) client implementations.

## Getting Started

Each module/assignment folder contains its own specific instructions and environment setup. Generally:

1. **Module 1**: Requires PyTorch and common data science libraries (`jupyter`, `matplotlib`, `requests`).
2. **Module 2**: Uses a `Makefile` for streamlined setup and execution. Run `make help` in the `module2-agents` directory to see available commands.

## Requirements

- Python 3.10+
- PyTorch
- (Optional) NVIDIA GPU for faster Transformer training in Module 1.

---
*Created as part of the Nebius Academy AI Performance Engineering curriculum.*
