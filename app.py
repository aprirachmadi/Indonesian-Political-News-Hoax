import streamlit as st
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd
import re
import string

@st.cache(allow_output_mutation=True)
def get_model():
    tokenizer = AutoTokenizer.from_pretrained('indobenchmark/indobert-base-p1')
    model = AutoModelForSequenceClassification.from_pretrained("rachmadiapri/IndoBERT-PoliticsHoaxDetection-base-p1", use_auth_token='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')
    return tokenizer,model


tokenizer,model = get_model()

user_input = st.text_area('Enter Text to Clasify')
button = st.button("Classify")

d = {
    
  1:'Hoax',
  0:'Non Hoax'
}

# removing breaking space
def remove_breaking_space(text):
  return text.replace('\xa0', ' ')

# removing spaces that appear more than once
def remove_unused_space(text):
  return re.sub(' +', ' ', text)

# remove text inside parentheses
def remove_parentheses(text):
  return re.sub("[\(\[].*?[\)\]]", "", text)

# turn text to lowercases
def lowercasing(text):
  return text.lower()

# give space for punctuation
def punctuation_spacing(text):
  for punct in string.punctuation:
    text = text.replace(punct, f' {punct} ')
  return re.sub(' +', ' ', text)

def preprocessing(text):
    text = remove_breaking_space(text)
    text = remove_unused_space(text)
    text = remove_parentheses(text)
    text = lowercasing(text)
    text = punctuation_spacing(text)
    return text


if user_input and button :
    user_input = preprocessing(user_input)
    test_sample = tokenizer([user_input], padding=True, truncation=True, max_length=512,return_tensors='pt')
    # test_sample
    output = model(**test_sample)
    st.write("Logits: ",output.logits)
    y_pred = np.argmax(output.logits.detach().numpy(),axis=1)
    st.write("Prediction: ",d[y_pred[0]])