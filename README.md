# Sentiment Analysis NLP System

This repository contains a comprehensive Natural Language Processing (NLP) system for sentiment classification. The project implements multiple approaches from classical machine learning to neural networks for analyzing sentiment in text data.

## Features

- **Multiple Model Implementations**:

  - Logistic Regression with customizable features
  - Deep Averaging Network (DAN) neural architecture
  - Baseline classifier for comparison

- **Text Processing**:

  - Advanced preprocessing pipeline
  - TF-IDF weighted unigram and bigram features
  - Stop word removal and frequency thresholding
  - Document frequency calculations

- **Word Embeddings**:

  - Integration with pre-trained GloVe embeddings
  - Vocabulary relativization to optimize memory usage
  - Support for fine-tuning embeddings during training

- **Evaluation Framework**:
  - Comprehensive metrics (Accuracy, Precision, Recall, F1)
  - Cross-validation with train/dev/test splits
  - Support for blind test set evaluation

## Setup and Installation

1. Clone the repository:

   ```
   git clone https://github.com/yourusername/sentiment-analysis.git
   cd sentiment-analysis
   ```

2. Install required dependencies:

   ```
   pip install numpy torch
   ```

3. Download GloVe embeddings:
   - Download from [Stanford NLP Group](https://nlp.stanford.edu/projects/glove/)
   - Place embeddings in the `data/` directory
   - Run `python sentiment_data.py` to relativize the embeddings to the dataset

## Project Structure

```
.
├── data/                   # Dataset files and word embeddings
├── sentiment_classifier.py # Main entry point
├── models.py               # Model implementations
├── sentiment_data.py       # Data loading and processing
├── utils.py                # Utility functions and classes
├── ffnn_example.py         # Example neural network implementation
└── Report.pdf              # Detailed project report
```

## Usage

### Basic Usage

To train and evaluate a model:

```
python sentiment_classifier.py --model [MODEL_TYPE]
```

Where `[MODEL_TYPE]` can be:

- `TRIVIAL`: Simple baseline that always predicts positive
- `LR`: Logistic Regression with TF-IDF features
- `DAN`: Deep Averaging Network

### Advanced Options

```
python sentiment_classifier.py --model DAN --word_vecs_path data/glove.6B.300d-relativized.txt --hidden_size 100 --lr 0.001 --num_epochs 10
```

Available parameters:

- `--train_path`: Path to training data (default: data/train.txt)
- `--dev_path`: Path to development data (default: data/dev.txt)
- `--blind_test_path`: Path to test data (default: data/test-blind.txt)
- `--word_vecs_path`: Path to word embeddings (default: data/glove.6B.300d-relativized.txt)
- `--lr`: Learning rate (default: 0.001)
- `--num_epochs`: Number of training epochs (default: 10)
- `--hidden_size`: Size of hidden layers for neural models (default: 100)
- `--batch_size`: Batch size for training (default: 1)
- `--feats`: Feature type for linear models (default: UNIGRAM)

## Model Performance

The system achieves solid performance on sentiment classification:

### Logistic Regression Model

- **Dev Accuracy**: 0.7947
- **Test Accuracy**: 0.8062

### Deep Averaging Network (DAN) Model

- **Dev Accuracy**: 0.8050
- **Test Accuracy**: 0.7853

The Logistic Regression model performs slightly better on the test set, while the DAN model shows slightly higher accuracy on the development set. This suggests that while neural approaches can be powerful, traditional ML approaches with strong feature engineering remain competitive for sentiment analysis tasks.

## Implementation Details

### Logistic Regression

The Logistic Regression model uses a variety of features including:

- TF-IDF weighted bag-of-words
- Bigram features
- Stop word removal
- Minimum frequency thresholding

### Deep Averaging Network

The DAN architecture:

1. Averages word embeddings for the input text
2. Passes the average through multiple feed-forward layers
3. Applies ReLU activation and dropout for regularization
4. Outputs a binary sentiment prediction

The model can be configured to either use frozen pre-trained embeddings or fine-tune them during training.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
