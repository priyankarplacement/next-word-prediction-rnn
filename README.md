# Next-Word Prediction using RNN

A simple text generation system built with a Recurrent Neural Network (RNN).
Course project (LLMs): given a seed phrase, the model predicts the next word and repeats this to generate text.

## Features
- Web scraping from Wikipedia (Space research article)
- Text preprocessing (lowercasing, citation removal, sentence splitting)
- N-gram sequence generation for training
- RNN model (Embedding + SimpleRNN + softmax) built with TensorFlow/Keras
- Autoregressive text generation from a user-entered seed phrase

## How It Works
1. Scrape the article text with `requests` and `BeautifulSoup`
2. Clean the text and split it into sentences
3. Tokenize words into integer IDs
4. Build growing n-gram sequences (the last word is the label)
5. Pad sequences and one-hot encode the labels
6. Train the model (categorical cross-entropy, Adam)
7. Generate text by predicting the most likely next word, appending it, and repeating

## Technologies Used
- Python
- TensorFlow / Keras
- BeautifulSoup
- NumPy
- Requests

## How to Run
```bash
pip install -r requirements.txt
python main.py
```
Then enter a seed phrase (for example `space exploration`) and the number of words to generate.

## Limitations
- Small dataset (500 sentences from one article)
- SimpleRNN forgets long-range context; an LSTM or GRU would do better
- Greedy decoding (argmax) makes the output repetitive
- Word-level tokenizer cannot handle unseen words

## Future Improvements
- Compare SimpleRNN with LSTM/GRU using validation loss
- Use temperature or top-k sampling for more varied text
- Train on a larger corpus and report perplexity
