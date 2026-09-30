# Sentiment Analysis using Transformers

A PyTorch implementation of a Transformer model trained from scratch for binary sentiment classification on the IMDB movie review dataset.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Project Objectives](#project-objectives)
- [Dataset](#dataset)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Training](#training)
- [Results](#results)
- [Key Takeaways](#key-takeaways)
- [Future Improvements](#future-improvements)

---

## Overview

This project demonstrates how to build, train, and evaluate a Transformer model from scratch using PyTorch for sentiment analysis. Unlike approaches that fine-tune pretrained models like BERT, this implementation constructs the entire Transformer architecture from the ground up, then adapts it for a **binary classification task** (positive vs. negative movie reviews).

The model is trained on the IMDB dataset and achieves **~78% test accuracy** on unseen reviews.

---

## Project Objectives

This project demonstrates competency in the following learning objectives:

- **Loading, exploring, and preparing** a text dataset for training a Transformer model using PyTorch
- **Customizing the architecture** of a Transformer model for a classification task
- **Training and testing** a Transformer model on the IMDB dataset
- Building a **custom PyTorch `Dataset`** and `DataLoader` with tokenization
- Implementing an **accuracy calculation method** for monitoring performance

---

## Dataset

The project uses the [IMDB dataset](https://ai.stanford.edu/~amaas/data/sentiment/), a standard benchmark for binary sentiment classification.

### Data Structure

```
aclIMDB/
├── train/
│   ├── pos/    # 12,500 positive reviews (label = 1)
│   ├── neg/    # 12,500 negative reviews (label = 0)
│   └── unsup/  # Unsupervised data (not used)
└── test/
    ├── pos/    # 12,500 positive reviews
    └── neg/    # 12,500 negative reviews
```

### Splits Used

| Split       | Size    | Source                                     |
|-------------|---------|--------------------------------------------|
| Training    | 22,500  | 90% of the original training set           |
| Validation  | 2,500   | 10% of the original training set           |
| Test        | 25,000  | Original test set                          |

Reviews are labeled as **1 (positive)** or **0 (negative)**.

---

## Architecture

The model is a custom, decoder-style Transformer adapted for classification. Key components:

### Components

1. **Token Embeddings** — Maps token IDs to dense vectors of size `d_embed = 256`
2. **Positional Embeddings** — Learned positional encodings of size `context_size = 128`
3. **Transformer Blocks** (×4), each containing:
   - **Multi-Head Attention** (8 heads × 32 dims)
   - **FeedForward Network** (GELU activation, 4× expansion)
   - **Layer Normalization** (Pre-LN) with residual connections
4. **Mean Pooling** — Aggregates token-level embeddings across the sequence dimension
5. **Classification Head** — Linear layer mapping `d_embed → num_classes (2)`

### Configuration

```python
config = {
    "vocabulary_size": tokenizer.vocab_size,  # ~30522 (bert-base-uncased)
    "num_classes": 2,
    "d_embed": 256,
    "context_size": 128,
    "layers_num": 4,
    "heads_num": 8,
    "head_size": 32,
    "dropout_rate": 0.1,
    "use_bias": True
}
```

### Tokenization

Uses the `bert-base-uncased` tokenizer from Hugging Face with:
- `truncation=True`
- `padding="max_length"`
- `max_length=128`

---

## Project Structure

```
.
├── Sentiment_Analysis_using_Transformers.ipynb   # Main notebook
├── aclImdb_v1.tar.gz                             # IMDB dataset archive
├── sentiment_scope_model_#1.pth                  # Saved model checkpoint
└── README.md
```

### Notebook Sections

1. **Introduction** — Overview, learning objectives, and sentiment analysis background
2. **Load, Explore, and Prepare the Dataset** — Loading reviews, EDA, and train/val split
3. **Implement a DataLoader in PyTorch** — `IMDBDataset` class + `DataLoader`
4. **Transformer Architecture** — Custom `DemoGPT` model and building blocks
5. **Accuracy Calculation Method** — Evaluation utility
6. **Train the Model** — Full training loop
7. **Test the Model** — Final evaluation + inference function
8. **Conclusion** — Results and takeaways

---

## Installation

### Requirements

- Python 3.10+
- PyTorch (with CUDA support recommended)
- Hugging Face Transformers
- pandas, matplotlib

### Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd sentiment-analysis-transformers

# Install dependencies
pip install torch transformers pandas matplotlib tqdm
```

### Dataset

Download the IMDB dataset (`aclImdb_v1.tar.gz`) from [Stanford AI Lab](https://ai.stanford.edu/~amaas/data/sentiment/) and place it in the project root.

---

## Usage

### Running the Notebook

```bash
jupyter notebook Sentiment_Analysis_using_Transformers.ipynb
```

Run all cells to:
1. Extract the dataset (`!tar -xzf aclImdb_v1.tar.gz`)
2. Preprocess and tokenize the reviews
3. Train the Transformer model
4. Evaluate test accuracy
5. Save the trained model

### Inference Example

```python
review = "This movie was absolutely fantastic. The acting was superb!"

result = predict_sentiment(review, model, tokenizer, device)
print("Predicted sentiment:", result)  # "Positive" or "Negative"
```

### Loading a Saved Model

```python
checkpoint = torch.load("sentiment_scope_model_#1.pth", map_location=device)
config = checkpoint["config"]
model = DemoGPT(config).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
```

---

## Training

### Hyperparameters

| Parameter      | Value     |
|----------------|-----------|
| Epochs         | 4         |
| Batch Size     | 32        |
| Optimizer      | AdamW     |
| Learning Rate  | 3e-4      |
| Loss Function  | CrossEntropyLoss |
| Max Seq Length | 128       |

### Training Loop

For each epoch:
1. Forward pass through the model
2. Compute cross-entropy loss
3. Backpropagate and update weights
4. Evaluate validation accuracy

---

## Results

### Training Progress

| Epoch | Validation Accuracy |
|-------|---------------------|
| 1     | 75.36%              |
| 2     | 78.48%              |
| 3     | 79.64%              |
| 4     | **80.20%**          |

### Final Test Performance

| Metric          | Value      |
|-----------------|------------|
| Test Accuracy   | **78.11%** |

The model exceeded the project's target of **75% accuracy** on the test set.

---

## Key Takeaways

1. **Text preprocessing and tokenization** are essential for NLP tasks.
2. **Transformers** can capture relationships between tokens effectively.
3. **PyTorch DataLoaders** simplify batch processing.
4. **Cross-Entropy Loss** is suitable for multi-class classification.
5. **Validation accuracy** helps monitor and diagnose model performance.
6. **Test accuracy** indicates generalization to unseen data.
7. **Hyperparameter tuning** and additional epochs can further improve performance.

---

## Future Improvements

- Increase training epochs and tune the learning rate schedule
- Experiment with different `d_embed`, `layers_num`, and `heads_num` values
- Replace mean pooling with a `[CLS]` token approach
- Add attention-mask support (padding-aware attention)
- Compare against fine-tuned pretrained models (BERT, RoBERTa)
- Add early stopping and learning rate warmup

---

## License

This project is for educational purposes. The IMDB dataset is provided by Stanford University for academic use.

---

## Acknowledgments

- [IMDB Dataset](https://ai.stanford.edu/~amaas/data/sentiment/) — Maas et al., 2011
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/) — tokenizer
- [PyTorch](https://pytorch.org/) — deep learning framework
- "Attention Is All You Need" — Vaswani et al., 2017 (original Transformer paper)