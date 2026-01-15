import numpy as np
import tensorflow as tf
import requests
from bs4 import BeautifulSoup
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# -------------------------------
# Web scraping
# -------------------------------
def extract_text_from_webpage(url):
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        raise Exception("Failed to load webpage")

    soup = BeautifulSoup(response.text, "html.parser")
    paragraphs = soup.find_all("p")
    text = " ".join(p.get_text() for p in paragraphs)
    return text


# -------------------------------
# Text preprocessing
# -------------------------------
def preprocess_text(text):
    text = text.lower()
    sentences = text.split(".")
    sentences = [s.strip() for s in sentences if len(s.strip()) > 0]
    return sentences


# -------------------------------
# Create training sequences
# -------------------------------
def create_sequences(texts, tokenizer):
    sequences = []
    for line in texts:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            sequences.append(token_list[:i + 1])
    return sequences


# -------------------------------
# Prepare model data
# -------------------------------
def prepare_data(sequences, total_words):
    max_len = max(len(seq) for seq in sequences)
    sequences = pad_sequences(sequences, maxlen=max_len, padding="pre")

    X = sequences[:, :-1]
    y = sequences[:, -1]
    y = tf.keras.utils.to_categorical(y, num_classes=total_words)

    return X, y, max_len


# -------------------------------
# Build RNN model
# -------------------------------
def create_model(total_words, max_len):
    model = Sequential([
        Embedding(total_words, 10, input_length=max_len - 1),
        SimpleRNN(64),
        Dense(total_words, activation="softmax")
    ])

    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"]
    )
    return model


# -------------------------------
# Text generation
# -------------------------------
def generate_text(seed_text, next_words, model, tokenizer, max_len):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences(
            [token_list], maxlen=max_len - 1, padding="pre"
        )

        predicted = model.predict(token_list, verbose=0)
        predicted_index = np.argmax(predicted, axis=-1)[0]

        for word, index in tokenizer.word_index.items():
            if index == predicted_index:
                seed_text += " " + word
                break

    return seed_text


# -------------------------------
# MAIN
# -------------------------------
def main():
    print("Scraping data...")
    url = "https://en.wikipedia.org/wiki/Space_research"
    raw_text = extract_text_from_webpage(url)

    texts = preprocess_text(raw_text)
    texts = texts[:500]  # IMPORTANT: limit data size

    print("Preparing tokenizer...")
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
    total_words = len(tokenizer.word_index) + 1

    print("Creating sequences...")
    sequences = create_sequences(texts, tokenizer)

    print("Preparing training data...")
    X, y, max_len = prepare_data(sequences, total_words)

    print("Building model...")
    model = create_model(total_words, max_len)

    print("Training model...")
    model.fit(X, y, epochs=30, verbose=1)

    print("\n--- TEXT GENERATION ---")
    seed = input("Enter seed text: ").lower()
    num_words = int(input("Number of words to generate: "))

    result = generate_text(seed, num_words, model, tokenizer, max_len)
    print("\nGenerated Text:")
    print(result)


if __name__ == "__main__":
    main()
