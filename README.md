# Text-classification---spam-or-ham
# Image-Retrieval
## **Overview**
This project will focus on constructing spam or ham mail classification application
The rest content of this article guide how to set up and run execution
Input: text
Output: whether the mail is spam or not (True of False)

### **Outlines**
This part will focus on:
* Install libraries
* Data preprocessing
* Train
* Run prediction

### *Prerequisites*
The following steps are for establishing environment:
1. Install python (3.10.14), you can download from this link:
[python.org](http://~https://www.python.org/downloads/)

2. Import libraries

```
import string
import nltk
nltk.download ('stopwords')
nltk.download ('punkt')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
```

3. Read data

Use pandas to load .csv file and extract data messages and label
```
DATASET_PATH = '/content/2cls_spam_text_cls.csv'
df = pd.read_csv(DATASET_PATH)

messages = df['Message'].values.tolist()
labels = df['Category'].values.tolist()
```
4. Data preprocessing

Data preprocessing plays a crucial role in LLM since it breaks down the text into small pieces called tokens to feed model.
Based on tokens, the model can learn meaning and context.

```
def lowercase(text):
  return text.lower()

def punctuation_removal(text):
  translator = str.maketrans('', '', string.punctuation)
  return text.translate(translator)

def tokenize(text):
  return nltk.word_tokenize(text)

def remove_stopwords(tokens):
  stop_words = nltk.corpus.stopwords.words('english')
  return [token for token in tokens if token not in stop_words]

def stemming(tokens):
  stemmer = nltk.PorterStemmer()

  return [stemmer.stem(token) for token in tokens]

def preprocess_text(text):
  text = lowercase(text)
  text = punctuation_removal(text)
  tokens = tokenize(text)
  tokens = remove_stopwords(tokens)
  tokens = stemming(tokens)
  return tokens

  messages = [preprocess_text(message) for message in messages]
```

5. Create dictionary

Tokens from input is nothing without dictionary. A dictionary is created to let the model knows the base of input
```
def create_dictionary(messages):
  dictionary = []
  for token in messages:
    if token not in dictionary:
      dictionary.append(token)

  return dictionary

dictionary = create_dictionary(messages)v
```

6. Create feature

```
def create_features(tokens, dictionary):
  features = np.zeros(len(dictionary))
  for token in tokens:
    if token in dictionary:
      features[dictionary.index(token)] += 1
  return features

X = np.array([create_features(tokens, dictionary) for tokens in messages])
```

7. Label feature

```
le = LabelEncoder()
y = le.fit_transform(labels)
print(f'Classes: {le.classes_}')
print(f'Encoded labels: {y}')
```

8. Seperate data for train, validation and test

```
VAL_SIZE = 0.2
TEST_SIZE = 0.125
SEED = 0

X_train, X_val, y_train, y_val = train_test_split(X, y,
                                                  test_size = VAL_SIZE,
                                                  shuffle=True,
                                                  random_state=SEED)

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train,
                                                  test_size = TEST_SIZE,
                                                  shuffle=True,
                                                  random_state=SEED)
```

9. Train model

```
model = GaussianNB ()
print("start training...")
model = model.fit(X_train, y_train)
print("finished ")
```

10. Run Prediction

```
def predict(text, model, dictionary):
  processed_text = preprocess_text(text)
  features = create_features(text, dictionary)
  features = np.array(features).reshape(1, -1)
  prediction = model.predict(features)
  prediction_cls = le.inverse_transform(prediction)[0]
  return prediction_cls

test_input = 'I am actually thinking a way of doing something useful'
preidction_cls = predict(test_input, model, dictionary)
print(f"Prediction{preidction_cls}")
```