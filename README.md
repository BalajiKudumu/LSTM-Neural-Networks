
# Long Short-Term Memory (LSTM) Neural Networks

## What is LSTM?

**LSTM (Long Short-Term Memory)** is a type of **Recurrent Neural Network (RNN)** designed to effectively learn long-term dependencies in sequence data. LSTMs are capable of remembering information over long periods because they address the *vanishing gradient problem* faced by traditional RNNs.

LSTMs are commonly used in:
- Natural Language Processing (NLP) (e.g., next word prediction, machine translation)
- Time series forecasting (e.g., stock prices, weather prediction)
- Speech recognition
- Music generation

---

## Why LSTM?

- Traditional RNNs struggle with long-term dependencies; they tend to "forget" distant information.
- LSTMs solve this using **memory cells** and **gates** that regulate information flow, helping the network retain important information longer.

---

## LSTM Architecture

An LSTM unit consists of:
- **Cell state (Ct)**: The memory of the network.
- **Hidden state (ht)**: The output at each time step.
- **Gates**: Structures that regulate what information is added, removed, or output.

---
## LSTM Cell Architecture
- An LSTM cell consists of:
- Forget Gate
- Input Gate
- Cell State Update
- Output Gate

🔹**Forget Gate**
Decides what information to discard from the cell state.


![LSTM Architecture](https://github.com/BalajiKudumu/LSTM-NeuralNetworks/blob/main/Forget_Gate.png?raw=true)


🔹 **Input Gate**
Determines which new information to store in the cell state.


![LSTM Architecture](https://github.com/BalajiKudumu/LSTM-NeuralNetworks/blob/main/Input_Gate.png?raw=true)


🔹 **Cell State Update**
Updates the cell state using the forget and input gates.

## Key Strengths of LSTM

- Allow selective memory and forgetting  
- Help LSTM learn long-term dependencies  
- Reduce vanishing gradient problems  
- Make LSTM effective for sequential data 

## This repository implements an LSTM Neural Network in Python for Next Word Prediction in Natural Language Processing (NLP) tasks.
