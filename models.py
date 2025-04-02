# models.py

import torch
import torch.nn as nn
from torch import optim
import numpy as np
import random
from typing import List
from sentiment_data import *
from utils import *
from collections import Counter


class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """

    def predict(self, ex_words: List[str]) -> int:
        """
        Makes a prediction on the given sentence
        :param ex_words: words to predict on
        :return: 0 or 1 with the label
        """
        raise Exception("Don't call me, call my subclasses")

    def predict_all(self, all_ex_words: List[List[str]]) -> List[int]:
        """
        You can leave this method with its default implementation, or you can override it to a batched version of
        prediction if you'd like. Since testing only happens once, this is less critical to optimize than training
        for the purposes of this assignment.
        :param all_ex_words: A list of all exs to do prediction on
        :return:
        """
        return [self.predict(ex_words) for ex_words in all_ex_words]


class TrivialSentimentClassifier(SentimentClassifier):
    def predict(self, ex_words: List[str]) -> int:
        """
        :param ex:
        :return: 1, always predicts positive class
        """
        return 1


class FeatureExtractor(object):
    """
    Feature extraction base type. Takes a sentence and returns an indexed list of features.
    """

    def get_indexer(self):
        raise Exception("Don't call me, call my subclasses")

    def extract_features(
        self, sentence: List[str], add_to_indexer: bool = False
    ) -> Counter:
        """
        Extract features from a sentence represented as a list of words. Includes a flag add_to_indexer to
        :param sentence: words in the example to featurize
        :param add_to_indexer: True if we should grow the dimensionality of the featurizer if new features are encountered.
        At test time, any unseen features should be discarded, but at train time, we probably want to keep growing it.
        :return: A feature vector. We suggest using a Counter[int], which can encode a sparse feature vector (only
        a few indices have nonzero value) in essentially the same way as a map. However, you can use whatever data
        structure you prefer, since this does not interact with the framework code.
        """
        raise Exception("Don't call me, call my subclasses")


class UnigramFeatureExtractor(FeatureExtractor):
    """
    Extracts unigram bag-of-words features from a sentence. It's up to you to decide how you want to handle counts
    and any additional preprocessing you want to do.
    """

    def __init__(self, indexer: Indexer, min_freq=2):
        self.indexer = (
            indexer  #    Keep track of words feature string maps to a numerical value
        )
        self.min_freq = min_freq
        self.word_freq = Counter()
        self.doc_freq = Counter()
        self.num_docs = 0
        self.stop_words = set(
            [
                "the",
                "a",
                "an",
                "and",
                "or",
                "but",
                "in",
                "on",
                "at",
                "to",
                "for",
                "is",
                "are",
                "was",
                "were",
            ]
        )

    def preprocess_word(self, word: str) -> str:
        """Preprocess a single word"""
        # Convert to lowercase and remove punctuation
        word = word.lower().strip(".,!?()[]{}\"'")
        return word

    def compute_idf(self):
        """Compute Inverse Document Frequency (IDF) Calculation values for all words
        - This is to show how rare a word is across all document

        - If a word appears in many documents (e.g., “the”), its doc_freq is high, so its IDF is low.

        - If a word appears in only a few documents (e.g., “amazing”), its doc_freq is low, so its IDF is high,
        meaning it’s more discriminative or “informative.”
        """
        # self.num_docs is the total number of documents
        # DF(word)= ln( num_docs / doc_freq(word)+1)

        self.idf = {
            word: np.log(self.num_docs / (freq + 1))
            for word, freq in self.doc_freq.items()
        }

    def initial_pass(self, all_examples: List[List[str]]):
        """First pass to compute word frequencies and document frequencies for all words.
            Secondly
        :param all_examples: list of all examples


        """
        for sentence in all_examples:
            word_presence = set()
            for word in sentence:
                word = self.preprocess_word(word)
                if word and word not in self.stop_words:
                    self.word_freq[word] += 1
                    word_presence.add(word)
            # Update document frequencies
            for word in word_presence:
                self.doc_freq[word] += 1
            self.num_docs += 1

        # Filter rare words and compute IDF
        self.valid_words = {
            word for word, count in self.word_freq.items() if count >= self.min_freq
        }
        self.compute_idf()

    def extract_features(
        self, sentence: List[str], add_to_indexer: bool = False
    ) -> Counter:
        features = Counter()

        # Get term frequencies,
        # TF is how many times a word appears in the document (relative to document length).
        # IDF is how rare (or important) that word is across all documents.
        term_freq = Counter()
        for word in sentence:
            word = self.preprocess_word(word)
            if word and word not in self.stop_words and word in self.valid_words:
                term_freq[word] += 1

        # Convert to Term Frequency-IDF features
        for word, tf in term_freq.items():
            # Get the index of the word from the Indexer
            index = self.indexer.add_and_get_index(word, add_to_indexer)
            if index != -1:  # Not seen before
                # TF-IDF calculation
                tfidf = (1 + np.log(tf)) * self.idf.get(word, 0)
                features[index] = tfidf

        return features

    def get_indexer(self):
        return self.indexer


class BigramFeatureExtractor(FeatureExtractor):
    """
    Bigram feature extractor analogous to the unigram one.
    """

    def __init__(self, indexer: Indexer):
        self.indexer = indexer
        self.min_freq = 2
        self.bigram_freq = Counter()
        self.doc_freq = Counter()
        self.num_docs = 0

    def initial_pass(self, all_examples: List[List[str]]):
        """First pass to compute bigram frequencies"""
        for sentence in all_examples:
            bigram_presence = set()  # Track unique bigrams in this document

            # Create bigrams from the sentence
            for i in range(len(sentence) - 1):
                bigram = f"{sentence[i]} {sentence[i + 1]}"
                self.bigram_freq[bigram] += 1
                bigram_presence.add(bigram)

            # Update document frequencies for bigrams in this document
            for bigram in bigram_presence:
                self.doc_freq[bigram] += 1
            self.num_docs += 1

        # Filter rare bigrams
        self.valid_bigrams = {
            bigram
            for bigram, count in self.bigram_freq.items()
            if count >= self.min_freq
        }

    def extract_features(
        self, sentence: List[str], add_to_indexer: bool = False
    ) -> Counter:
        features = Counter()

        # Extract bigram features
        for i in range(len(sentence) - 1):
            bigram = f"{sentence[i]} {sentence[i + 1]}"
            if bigram in self.valid_bigrams or add_to_indexer:
                index = self.indexer.add_and_get_index(bigram, add_to_indexer)
                if index != -1:
                    features[index] += 1

        return features

    def get_indexer(self):
        return self.indexer


class BetterFeatureExtractor(FeatureExtractor):
    """
    Better feature extractor...try whatever you can think of!
    1. Unigrams and bigrams
    2. Sentiment-bearing word features
    3. Negation handling
    """

    def __init__(self, indexer: Indexer):
        self.positive_words = {
            "good",
            "great",
            "awesome",
            "nice",
            "excellent",
            "amazing",
            "wonderful",
            "best",
            "fantastic",
            "perfect",
            "brilliant",
            "outstanding",
            "superb",
            "terrific",
        }

        self.negative_words = {
            "bad",
            "terrible",
            "awful",
            "worst",
            "horrible",
            "poor",
            "disappointing",
            "sucks",
            "disgusting",
            "lousy",
            "pathetic",
            "crap",
            "shitty",
        }
        self.indexer = indexer
        self.word_freq = Counter()
        self.doc_freq = Counter()
        self.num_docs = 0
        self.min_freq = 2

    def initial_pass(self, all_examples: List[List[str]]):
        """First pass to compute word frequencies and document frequencies"""
        for sentence in all_examples:
            word_presence = set()  # Track unique words in this document

            # Count word frequencies
            for word in sentence:
                word = word.lower().strip()
                self.word_freq[word] += 1
                word_presence.add(word)

            # Update document frequencies
            for word in word_presence:
                self.doc_freq[word] += 1
            self.num_docs += 1

        # Filter rare words
        self.valid_words = {
            word for word, count in self.word_freq.items() if count >= self.min_freq
        }

    def extract_features(
        self, sentence: List[str], add_to_indexer: bool = False
    ) -> Counter:
        features = Counter()

        # Extract unigram features
        for word in sentence:
            word = word.lower().strip()
            if word in self.valid_words or add_to_indexer:
                unigram_idx = self.indexer.add_and_get_index(
                    f"UNIGRAM_{word}", add_to_indexer
                )
                if unigram_idx != -1:
                    features[unigram_idx] += 1

        # Extract bigram features
        for i in range(len(sentence) - 1):
            bigram = f"{sentence[i]} {sentence[i + 1]}".lower()
            bigram_idx = self.indexer.add_and_get_index(
                f"BIGRAM_{bigram}", add_to_indexer
            )
            if bigram_idx != -1:
                features[bigram_idx] += 1

        # Add sentiment word features
        pos_count = sum(1 for word in sentence if word.lower() in self.positive_words)
        neg_count = sum(1 for word in sentence if word.lower() in self.negative_words)

        pos_idx = self.indexer.add_and_get_index("POSITIVE_WORDS", add_to_indexer)
        neg_idx = self.indexer.add_and_get_index("NEGATIVE_WORDS", add_to_indexer)

        if pos_idx != -1:
            features[pos_idx] = pos_count
        if neg_idx != -1:
            features[neg_idx] = neg_count

        return features

    def get_indexer(self):
        return self.indexer


class LogisticRegressionClassifier(SentimentClassifier):
    """
    Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
    superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
    modify the constructor to pass these in.
    """

    def __init__(self, learning_rate=0.01, num_epochs=1000, lambda_reg=0.0):
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.lambda_reg = lambda_reg
        self.weights = None
        self.bias = None
        self.feat_extractor = None

    def sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        input_size, input_features = X.shape
        self.weights = np.zeros(input_features)  # Initialize to zeros instead of random
        self.bias = 0.0

        # Use mini-batches for better stability
        batch_size = 32
        n_batches = input_size // batch_size

        for epoch in range(self.num_epochs):
            # Shuffle for SGD
            indices = np.arange(input_size)
            np.random.shuffle(indices)
            X = X[indices]
            y = y[indices]

            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = start_idx + batch_size

                X_batch = X[start_idx:end_idx]
                y_batch = y[start_idx:end_idx]

                y_pred = self.sigmoid(np.dot(X_batch, self.weights) + self.bias)

                # Compute gradients
                dw = (1 / batch_size) * np.dot(X_batch.T, (y_pred - y_batch))
                db = (1 / batch_size) * np.sum(y_pred - y_batch)

                # Add L2 regularization
                dw += self.lambda_reg * self.weights

                # Update weights and bias
                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db

    def predict(self, ex_words: List[str]) -> int:

        features = self.feat_extractor.extract_features(ex_words, add_to_indexer=False)
        vocab_size = len(self.feat_extractor.get_indexer())
        X = np.zeros(vocab_size)
        for idx, value in features.items():
            if idx < vocab_size:
                X[idx] = value

        y_pred = self.sigmoid(np.dot(X, self.weights) + self.bias)
        return 1 if y_pred > 0.5 else 0


def train_logistic_regression(
    train_exs: List[SentimentExample], feat_extractor: FeatureExtractor
) -> LogisticRegressionClassifier:
    """
    Train a logistic regression model.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained LogisticRegressionClassifier model
    """
    # Use the feature extractor to get the feature vectors for each example
    all_words = [ex.words for ex in train_exs]
    feat_extractor.initial_pass(all_words)

    # Extract features
    features_list, labels = [], []
    for ex in train_exs:
        features = feat_extractor.extract_features(ex.words, add_to_indexer=True)
        features_list.append(features)
        labels.append(ex.label)

    # Convert to numpy arrays
    vocab_size = len(feat_extractor.get_indexer())
    X = np.zeros((len(features_list), vocab_size))
    for i, features in enumerate(features_list):
        for idx, value in features.items():
            X[i, idx] = value
    y = np.array(labels)

    # Train classifier
    classifier = LogisticRegressionClassifier(
        learning_rate=0.1, num_epochs=100, lambda_reg=0.01
    )
    classifier.feat_extractor = feat_extractor
    classifier.fit(X, y)

    return classifier


def train_linear_model(
    args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample]
) -> SentimentClassifier:
    """
    Main entry point for your linear model. You may modify this, but do not need to.
    :param args: args bundle from sentiment_classifier.py
    :param train_exs: training set, List of SentimentExample objects
    :param dev_exs: dev set, List of SentimentExample objects. You can use this for validation throughout the training
    process, but you should *not* directly train on this data.
    :return: trained SentimentClassifier model, of whichever type is specified
    """
    # Initialize feature extractor
    if args.model == "TRIVIAL":
        feat_extractor = None
    elif args.feats == "UNIGRAM":
        # Add additional preprocessing code here
        feat_extractor = UnigramFeatureExtractor(Indexer())
    elif args.feats == "BIGRAM":
        # Add additional preprocessing code here
        feat_extractor = BigramFeatureExtractor(Indexer())
    elif args.feats == "BETTER":
        # Add additional preprocessing code here
        feat_extractor = BetterFeatureExtractor(Indexer())
    else:
        raise Exception(
            "Pass in UNIGRAM, BIGRAM, or BETTER to run the appropriate system"
        )

    # Train the model
    model = train_logistic_regression(train_exs, feat_extractor)
    return model


class NeuralSentimentClassifier(SentimentClassifier):
    """
    Implement your NeuralSentimentClassifier here. This should wrap an instance of the network with learned weights
    along with everything needed to run it on new data (word embeddings, etc.)
    """

    def __init__(self, network, word_embeddings):
        self.network = network
        self.word_embeddings = word_embeddings

    def predict(self, ex_words: List[str]) -> int:
        """
        Makes a prediction on the given sentence
        """
        self.network.eval()

        with torch.no_grad():
            # Convert words to indices
            word_indices = [
                self.word_embeddings.word_indexer.index_of(word.lower())
                for word in ex_words
            ]
            # Filter out unknown words
            word_indices = [
                idx if idx != -1 else self.word_embeddings.word_indexer.index_of("UNK")
                for idx in word_indices
            ]
            # Convert to tensor
            if len(word_indices) == 0:
                word_indices = [self.word_embeddings.word_indexer.index_of("UNK")]

            # Add batch dimension
            indices_tensor = torch.tensor(word_indices, dtype=torch.long).unsqueeze(0)

            # Get prediction
            log_probs = self.network(indices_tensor)

            # Return the most likely class
            return int(torch.argmax(log_probs, dim=1).item())


class DeepAveragingNetwork(nn.Module):
    """
    Deep Averaging Network for sentiment classification.
    Architecture:
    1. Embedding layer
    2. Average pooling of embeddings
    3. Feedforward layers with ReLU activation
    4. Output layer with log softmax
    """

    def __init__(self, embedding_layer, hidden_size, num_classes, dropout_rate=0.5):
        super(DeepAveragingNetwork, self).__init__()
        self.embed = embedding_layer
        self.embed.padding_idx = 0
        embedding_dim = embedding_layer.embedding_dim

        # Network architecture
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, num_classes),
        )

        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, word_indices):
        # Get embeddings for all words
        embeddings = self.embed(word_indices)
        mask = (word_indices != 0).float().unsqueeze(-1)

        # Masked average pooling
        masked_embeddings = embeddings * mask
        sum_embeddings = torch.sum(masked_embeddings, dim=1)
        seq_lengths = torch.sum(mask, dim=1)
        avg_embeddings = sum_embeddings / seq_lengths

        # Pass through the network
        scores = self.net(avg_embeddings)

        return self.log_softmax(scores)


def create_batches(
    examples: List[SentimentExample], batch_size: int, word_embeddings: WordEmbeddings
):
    """Helper function to create batches from examples"""
    batches = []
    for i in range(0, len(examples), batch_size):
        batch_examples = examples[i : i + batch_size]

        # Get max length in this batch
        max_len = max(len(ex.words) for ex in batch_examples)

        # Initialize tensors for this batch
        batch_indices = []
        batch_labels = []

        for ex in batch_examples:
            # Convert words to indices with padding
            indices = [
                word_embeddings.word_indexer.index_of(word.lower()) for word in ex.words
            ]
            indices = [
                idx if idx != -1 else word_embeddings.word_indexer.index_of("UNK")
                for idx in indices
            ]

            # Pad sequence
            padding = [0] * (max_len - len(indices))
            indices.extend(padding)

            batch_indices.append(indices)
            batch_labels.append(ex.label)

        # Convert to tensors
        batch_indices = torch.tensor(batch_indices, dtype=torch.long)
        batch_labels = torch.tensor(batch_labels, dtype=torch.long)

        batches.append((batch_indices, batch_labels))

    return batches


def train_deep_averaging_network(
    args,
    train_exs: List[SentimentExample],
    dev_exs: List[SentimentExample],
    word_embeddings: WordEmbeddings,
) -> NeuralSentimentClassifier:
    """
    Main entry point for your deep averaging network model.
    :param args: Command-line args so you can access them here
    :param train_exs: training examples
    :param dev_exs: development set, in case you wish to evaluate your model during training
    :param word_embeddings: set of loaded word embeddings
    :return: A trained NeuralSentimentClassifier model
    """
    # Create the embedding layer
    emb_layer = word_embeddings.get_initialized_embedding_layer(frozen=False)

    # Initialize model
    model = DeepAveragingNetwork(emb_layer, args.hidden_size, num_classes=2)

    # Loss function and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    best_dev_acc = 0.0
    best_model = None

    for epoch in range(2):
        model.train()
        total_loss = 0.0

        # Training loop
        for ex in train_exs:
            word_indices = [
                word_embeddings.word_indexer.index_of(word.lower()) for word in ex.words
            ]
            word_indices = [
                idx if idx != -1 else word_embeddings.word_indexer.index_of("UNK")
                for idx in word_indices
            ]

            if len(word_indices) == 0:
                word_indices = [word_embeddings.word_indexer.index_of("UNK")]

            indices_tensor = torch.tensor(word_indices, dtype=torch.long).unsqueeze(0)
            label_tensor = torch.tensor([ex.label], dtype=torch.long)

            model.zero_grad()
            log_probs = model(indices_tensor)
            loss = criterion(log_probs, label_tensor)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")

    return NeuralSentimentClassifier(model, word_embeddings)


def train_deep_averaging_network_batching(
    args,
    train_exs: List[SentimentExample],
    dev_exs: List[SentimentExample],
    word_embeddings: WordEmbeddings,
) -> NeuralSentimentClassifier:
    """
    Main entry point for your deep averaging network model.
    :param args: Command-line args so you can access them here
    :param train_exs: training examples
    :param dev_exs: development set, in case you wish to evaluate your model during training
    :param word_embeddings: set of loaded word embeddings
    :return: A trained NeuralSentimentClassifier model
    """
    # Create the embedding layer
    emb_layer = word_embeddings.get_initialized_embedding_layer(frozen=False)

    # Initialize model
    model = DeepAveragingNetwork(emb_layer, args.hidden_size, num_classes=2)

    # Loss function and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    best_dev_acc = 0.0
    best_model = None
    patience = 3
    no_improvement = 0

    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0.0

        # Create batches
        random.shuffle(train_exs)
        train_batches = create_batches(train_exs, args.batch_size, word_embeddings)

        for batch_indices, batch_labels in train_batches:
            # Forward pass
            model.zero_grad()
            log_probs = model(batch_indices)
            loss = criterion(log_probs, batch_labels)

            # Backward pass
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Evaluate on dev set with batching
        model.eval()
        dev_batches = create_batches(dev_exs, args.batch_size, word_embeddings)
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_indices, batch_labels in dev_batches:
                log_probs = model(batch_indices)
                predictions = torch.argmax(log_probs, dim=1)
                correct += (predictions == batch_labels).sum().item()
                total += len(batch_labels)

        dev_acc = correct / total
        print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}, Dev Accuracy: {dev_acc:.4f}")

        # Save best model and early stopping logic
        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            best_model = DeepAveragingNetwork(
                emb_layer, args.hidden_size, num_classes=2
            )
            best_model.load_state_dict(model.state_dict())
            no_improvement = 0
        else:
            no_improvement += 1

        if no_improvement >= patience:
            print(f"Early stopping after {epoch + 1} epochs")
            break

    return NeuralSentimentClassifier(best_model, word_embeddings)
