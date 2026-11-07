"""
Next-Word Prediction using RNN
Course project (LLMs): scrapes a Wikipedia article, builds n-gram training
sequences, trains an Embedding + SimpleRNN + softmax model, and generates text
from a seed phrase.
"""

import re

import numpy as np
import requests
import tensorflow as tf
from bs4 import BeautifulSoup
from tensorflow.keras.layers import Dense, Embedding, SimpleRNN
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

URL = "https://en.wikipedia.org/wiki/Space_research"
MAX_SENTENCES = 500   # limit data size: one-hot labels use a lot of memory
EPOCHS = 30
EMBEDDING_DIM = 10
RNN_UNITS = 64
VALIDATION_SPLIT = 0.2
SEED = 42


# -------------------------------
# Web scraping
# -------------------------------
def extract_text_from_webpage(url):
    """Download a page and return the text of all <p> tags."""
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers, timeout=15)

    if response.status_code != 200:
        raise RuntimeError(f"Failed to load webpage (status {response.status_code})")

    soup = BeautifulSoup(response.text, "html.parser")
    paragraphs = soup.find_all("p")
    return " ".join(p.get_text() for p in paragraphs)


# -------------------------------
# Text preprocessing
# -------------------------------
def preprocess_text(text):
    """Lowercase, remove citation marks like [12], and split into sentences."""
    text = text.lower()
    text = re.sub(r"\[\d+\]", "", text)      # remove Wikipedia citation numbers
    sentences = text.split(".")
    return [s.strip() for s in sentences if s.strip()]


# -------------------------------
# Create training sequences
# -------------------------------
def create_sequences(texts, tokenizer):
    """Turn each sentence into growing n-gram prefixes.

    'space research is important' ->
        [space, research]
        [space, research, is]
        [space, research, is, important]
    The last word of each prefix is the label to predict.
    """
    sequences = []
    for line in texts:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            sequences.append(token_list[: i + 1])
    return sequences


# -------------------------------
# Prepare model data
# -------------------------------
def prepare_data(sequences, total_words):
    """Pad sequences, split into inputs (X) and labels (y), one-hot encode y."""
    max_len = max(len(seq) for seq in sequences)
    sequences = pad_sequences(sequences, maxlen=max_len, padding="pre")

    # Shuffle so the validation split is not just the last sentences
    rng = np.random.default_rng(SEED)
    sequences = sequences[rng.permutation(len(sequences))]

    X = sequences[:, :-1]
    y = sequences[:, -1]
    y = tf.keras.utils.to_categorical(y, num_classes=total_words)
    return X, y, max_len


# -------------------------------
# Build RNN model
# -------------------------------
def create_model(total_words):
    model = Sequential([
        Embedding(total_words, EMBEDDING_DIM),
        SimpleRNN(RNN_UNITS),
        Dense(total_words, activation="softmax"),
    ])
    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"],
    )
    return model


# -------------------------------
# Text generation
# -------------------------------
def generate_text(seed_text, next_words, model, tokenizer, max_len):
    """Autoregressive generation: predict a word, append it, repeat."""
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        if not token_list:
            print("Warning: none of the seed words are in the vocabulary.")
            break

        token_list = pad_sequences([token_list], maxlen=max_len - 1, padding="pre")
        predicted = model.predict(token_list, verbose=0)
        predicted_index = int(np.argmax(predicted, axis=-1)[0])  # greedy decoding

        word = tokenizer.index_word.get(predicted_index)
        if word is None:
            break
        seed_text += " " + word

    return seed_text


# -------------------------------
# MAIN
# -------------------------------
def main():
    print("Scraping data...")
    raw_text = extract_text_from_webpage(URL)

    texts = preprocess_text(raw_text)[:MAX_SENTENCES]
    print(f"Using {len(texts)} sentences")

    print("Preparing tokenizer...")
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
    total_words = len(tokenizer.word_index) + 1  # +1 for the padding index 0
    print(f"Vocabulary size: {total_words}")

    print("Creating sequences...")
    sequences = create_sequences(texts, tokenizer)

    print("Preparing training data...")
    X, y, max_len = prepare_data(sequences, total_words)
    print(f"Training samples: {len(X)}, input length: {X.shape[1]}")

    print("Building model...")
    model = create_model(total_words)

    print("Training model...")
    model.fit(X, y, epochs=EPOCHS, validation_split=VALIDATION_SPLIT, verbose=1)

    print("\n--- TEXT GENERATION ---")
    seed = input("Enter seed text (e.g. 'space exploration'): ").lower()
    try:
        num_words = int(input("Number of words to generate: "))
    except ValueError:
        num_words = 5
        print("Invalid number, using 5.")

    result = generate_text(seed, num_words, model, tokenizer, max_len)
    print("\nGenerated Text:")
    print(result)


if __name__ == "__main__":
    main()
