import streamlit as st
import numpy as np
import pickle
import tensorflow
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.sequence import pad_sequences


## Load the LSTM model
model=tensorflow.keras.models.load_model('lstm_next_word.h5')

### Load the tokenizer
with open('tokenizer.pickle','rb') as file:
    tokenizer=pickle.load(file)



## Function to predict the next word
def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list=tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence_len:
        token_list=token_list[-(max_sequence_len-1):] ## Ensure the sequence length matches max_sequence_len
    token_list=tensorflow.keras.preprocessing.sequence.pad_sequences([token_list], maxlen=max_sequence_len, padding='pre')
    predicted=model.predict(token_list, verbose=0)
    predicted_word_index=np.argmax(predicted, axis=1)
    for word, index in tokenizer.word_index.items():
        if index== predicted_word_index:
            return word
    
    return None

st.title('Predicting next word with LSTM and Early Stopping')
input_text=st.text_input('Enter the sequence of words',"To be or not to be")
if st.button("Predict Next Word"):
    max_sequence_len=model.input_shape[1]+1   ## Retrive the max sequence length from the model input shape
    next_word=predict_next_word(model, tokenizer,input_text, max_sequence_len)
    st.write(f'Next word Predicted as: {next_word}')






