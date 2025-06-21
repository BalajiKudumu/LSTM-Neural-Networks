# Next_Word_Prediction_LSTM_RNN

## LSTM Neural Network for Next Word Prediction

## Overview

This repository explains how an **LSTM (Long Short-Term Memory)** neural network works and how it can be applied for **next-word prediction**.

LSTMs are a type of Recurrent Neural Network (RNN) designed to model long-range dependencies in sequential data, such as text.

---

## 🧠 LSTM Architecture

An LSTM unit consists of:

* **Cell state (c\_t)** — memory of the network
* **Hidden state (h\_t)** — output at time `t`
* **Gates**:

  * Forget gate `f_t`: What to discard
  * Input gate `i_t`: What to update
  * Output gate `o_t`: What to output

### LSTM Equations

Given input `x_t` at time `t`:

``
f_t = σ(W_f · [h_{t-1}, x_t] + b_f)

i_t = σ(W_i · [h_{t-1}, x_t] + b_i)

c̃_t = tanh(W_c · [h_{t-1}, x_t] + b_c)

c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t

o_t = σ(W_o · [h_{t-1}, x_t] + b_o)

h_t = o_t ⊙ tanh(c_t)
``

Where:

* `σ` = sigmoid activation
* `tanh` = hyperbolic tangent activation
* `⊙` = element-wise multiplication
* `W_f, W_i, W_c, W_o` = learned weights
* `b_f, b_i, b_c, b_o` = biases

--

## ✉️ Next-Word Prediction Example

**Input sequence:**

``
I love to
``

**Goal:** Predict next word.

1️⃣ Convert words to embeddings → feed to LSTM.

2️⃣ LSTM processes sequence → outputs hidden state `h_t`.

3️⃣ Dense + Softmax layer gives probability distribution over vocabulary.

### Example output (top 3 predictions):

| Next word candidates | Probability |
| -------------------- | ----------- |
| eat                  | 0.7         |
| play                 | 0.2         |
| run                  | 0.1         |

⚠️ **Note:** The model produces a full probability vector across the entire vocabulary. We often show top N words for readability or use in decoding strategies.

---

## ✅ Example Code (Keras)

``python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=64),
    LSTM(128),
    Dense(vocab_size, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam')
``

---

## ⚡ Summary

* **LSTM** remembers long-term context using gates.
* Outputs hidden states `h_t` that represent the sequence so far.
* A softmax layer predicts next word probability distribution.
* We can use the top prediction or sample from top-k candidates for next-word generation.

---

## 📂 Example usage

``python
input_text = ['I', 'love', 'to']
# Convert to indices or embeddings → feed to model → get softmax output
# Pick word with highest probability or sample
``

--

## Future improvements



--


