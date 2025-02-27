import random as random
from collections import Counter
import pandas as pd
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None) 
import os
os.chdir(r'C:\Users\Klaws\Downloads\ML_Crypto Research')
import nltk
nltk.download('punkt_tab') # text parser
import re
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
#%% 


# %% custom functions

# text cleaning
def remove_non_letters(text):
    return re.sub(r'[^A-Za-z\s]', '', text)  # Only keep letters and spaces


# encryption via shift
def shift_cipher(plaintext, key):
    ciphertext = ""
    plaintext = remove_non_letters(plaintext)
    plaintext = plaintext.replace(" ", "")
    plaintext = plaintext.lower()
    for char in plaintext:
        ciphertext += chr((ord(char) + key - 97) % 26 + 97)
        
    ciphertext = ciphertext.upper()

    return plaintext, ciphertext 


# decryption via unshift
def decrypt_with_shift(ciphertext, shift):
    decrypted_text = []
    for char in ciphertext:
        if 'A' <= char <= 'Z':  # Check if char is an uppercase letter
            decrypted_char = chr(((ord(char) - ord('A') - shift) % 26) + ord('A'))
            decrypted_text.append(decrypted_char)
    return ''.join(decrypted_text).lower()


# letter frequency counts
def char_frequency(text):
    counts = Counter(text)
    total_chars = len(text)
    return {char: count / total_chars for char, count in counts.items()}


# score text based on letter frequencies
def score_text(text):
    english_letter_freq = {
        'E': 12.70, 'T': 9.06, 'A': 8.17, 'O': 7.51, 'I': 6.97, 'N': 6.75, 
        'S': 6.33, 'H': 6.09, 'R': 5.99, 'D': 4.25, 'L': 4.03, 'C': 2.78, 
        'U': 2.76, 'M': 2.41, 'W': 2.36, 'F': 2.23, 'G': 2.02, 'Y': 1.97, 
        'P': 1.93, 'B': 1.49, 'V': 0.98, 'K': 0.77, 'J': 0.15, 'X': 0.15, 
        'Q': 0.10, 'Z': 0.07
    }
    text_counter = Counter(text.upper())
    text_len = len(text)
    score = 0 # similar to dot product (cosine similarity) in for loop calc
    for letter, count in text_counter.items():
        if letter in english_letter_freq:
            score += (count / text_len) * english_letter_freq[letter]
    return score


# break shift cypher
def crack_shift_cipher_auto(ciphertext):    
    # Try all possible shifts and score the resulting plaintexts
    best_shift = 0
    best_score = -1
    best_plaintext = ""
    
    for shift in range(26):
        decrypted = decrypt_with_shift(ciphertext, shift)
        current_score = score_text(decrypted)
        if current_score > best_score:
            best_score = current_score
            best_shift = shift
            best_plaintext = decrypted
    
    return best_shift, best_plaintext

# %% Text Preprocessing

# Split the text into chunks of 100 words
def split_text_into_chunks_by_words(text, chunk_size=100):
    words = nltk.word_tokenize(text)  # Tokenize into words
    chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks
# %% read in text file

with open('StarWars1.txt', 'r', encoding='utf-8') as file:
    chapter_text = file.read()

chunks = split_text_into_chunks_by_words(chapter_text, chunk_size=100)

# %% encrypt and store sentences

plain_cipher_pairs = []
for chunk in chunks:
    key = random.randint(1, 25)  # Random key for shift cipher
    plaintext, ciphertext = shift_cipher(chunk, key)
    plain_cipher_pairs.append((key, plaintext, ciphertext))

df = pd.DataFrame(plain_cipher_pairs, columns=['Key', 'Plaintext', 'Ciphertext'])
df = df.iloc[:-1].copy()

# %% full features for ML model training

df_ml_dataset = df.copy()
df_ml_dataset['Ciphertext_Freq'] = df_ml_dataset['Ciphertext'].apply(char_frequency)
df_ml_dataset = pd.DataFrame(df_ml_dataset['Ciphertext_Freq'].to_list()).fillna(0)
df_ml_dataset = df_ml_dataset[sorted(df_ml_dataset.columns)]
df_ml_dataset = pd.concat([df[['Key', 'Plaintext', 'Ciphertext']], df_ml_dataset.add_prefix('Cipher_')], axis=1)
print(f"Generated {len(df)} encrypted chunks with Shift cipher.\n")
# %% cryptanalysis by comparison to english letter frequency

df_cryptanalysis = df.copy()

df_cryptanalysis['best_key'] = None
df_cryptanalysis['best_plaintext'] = None

for i in range(df_cryptanalysis.shape[0]):
    ciphertext = df_cryptanalysis.loc[i, 'Ciphertext']
    best_key, best_plaintext = crack_shift_cipher_auto(ciphertext)
    
    df_cryptanalysis.loc[i, 'best_key'] = best_key
    df_cryptanalysis.loc[i, 'best_plaintext'] = best_plaintext
    
df_cryptanalysis['best_key'] = pd.to_numeric(df_cryptanalysis['best_key'], errors='coerce')
    
df_cryptanalysis['keys_match'] = df_cryptanalysis['Key'] == df_cryptanalysis['best_key']


print("Manual Cryptanalysis Scores: \n")
false_count = len(df_cryptanalysis) - df_cryptanalysis['keys_match'].sum()
print(f"CRYPTANALYSIS: Number of non-matching keys: {false_count}")
acc_perc = (len(df_cryptanalysis)-false_count)/len(df_cryptanalysis)
print(f"CRYPTANALYSIS: Percent of matching keys: {acc_perc:.4f}\n")

# %% ML model train-test

# Prepare the dataset
df_ml_dataset = df.copy()
df_ml_dataset['Ciphertext_Freq'] = df_ml_dataset['Ciphertext'].apply(char_frequency)
df_ml_dataset = pd.DataFrame(df_ml_dataset['Ciphertext_Freq'].to_list()).fillna(0)
df_ml_dataset = df_ml_dataset[sorted(df_ml_dataset.columns)]
df_ml_dataset = pd.concat([df[['Key', 'Plaintext', 'Ciphertext']], df_ml_dataset.add_prefix('Cipher_')], axis=1)

# Split dataset into training and test sets
X = df_ml_dataset.iloc[:, 3:]  # Features: all columns except 'Key'
y = df_ml_dataset['Key']  # Target: 'Key' (cipher key)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on the training and testing sets
train_preds = model.predict(X_train)
test_preds = model.predict(X_test)

# Round the predictions to the nearest integer (cipher key)
train_preds_rounded = np.round(train_preds).astype(int)
test_preds_rounded = np.round(test_preds).astype(int)

# Clip the predictions to ensure they stay within the valid cipher key range (1 to 25)
train_preds_clipped = np.clip(train_preds_rounded, 1, 25)
test_preds_clipped = np.clip(test_preds_rounded, 1, 25)

# Calculate accuracy using accuracy_score
LRtrain_accuracy = accuracy_score(y_train, train_preds_clipped)
LRtest_accuracy = accuracy_score(y_test, test_preds_clipped)

# Calculate R² (R-squared) for train and test
train_r2 = model.score(X_train, y_train)
test_r2 = model.score(X_test, y_test)

#%%
print("-----------------------------------------------------------------------\n")
#%%

print("These are the results for the Shift Cipher and Linear Regression\n")

print("Linear Regression Scores: \n")

print(f"Train Accuracy: {LRtrain_accuracy:.4f}")
print(f"Test Accuracy: {LRtest_accuracy:.4f}")
print(f"Train R²: {train_r2:.4f}")
print(f"Test R²: {test_r2:.4f}\n")

kf = KFold(n_splits=10, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=kf, scoring='r2')  # R² score for regression
print("Cross Validation Scores: \n")
print("Cross-validation R² scores:\n", scores)
print("Mean R² score:\n", np.mean(scores), "\n")

LRCrossVal = np.mean(scores)
#%%
print("-----------------------------------------------------------------------\n")
#%%

print("These are the results for the Shift Cipher and Logistic Regression\n")

# %% 
# Use Logistic Regression (classification model)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict on the training and testing sets
train_preds = model.predict(X_train)
test_preds = model.predict(X_test)

# Calculate accuracy using accuracy_score
LLRtrain_accuracy = accuracy_score(y_train, train_preds)
LLRtest_accuracy = accuracy_score(y_test, test_preds)
print("Logistic Regression Scores: \n")
print(f"Train Accuracy: {LLRtrain_accuracy:.4f}")
print(f"Test Accuracy: {LLRtest_accuracy:.4f}\n")

# K-Fold Cross-Validation (classification accuracy)
kf = KFold(n_splits=10, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')
print("Cross Validation Scores: \n")
print("Cross-validation scores:", scores)
print("Mean accuracy:\n", np.mean(scores), "\n")
LLRCrossVal = np.mean(scores)


#%%
print("-----------------------------------------------------------------------\n")
#%%

print("These are the results for the Shift Cipher and Decision Trees\n")

#%%

# %% 

# Use Decision Tree Classifier
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict on the training and testing sets
train_preds = model.predict(X_train)
test_preds = model.predict(X_test)

# Calculate accuracy using accuracy_score
DTtrain_accuracy = accuracy_score(y_train, train_preds)
DTtest_accuracy = accuracy_score(y_test, test_preds)
print("Decision Tree Scores: \n")
print(f"Train Accuracy: {DTtrain_accuracy:.4f}")
print(f"Test Accuracy: {DTtest_accuracy:.4f}\n")

# K-Fold Cross-Validation (classification accuracy)
kf = KFold(n_splits=10, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')
print("Cross Validation Scores: \n")
print("Cross-validation scores:", scores)
print("Mean accuracy:\n", np.mean(scores), "\n")
DTCrossVal = np.mean(scores)
#%%
print("-----------------------------------------------------------------------\n")
#%%

print("These are the results for the Shift Cipher and Random Forest \n")

#%%

# %% Random Forest

model = RandomForestClassifier(n_estimators=100, random_state=42)  # You can adjust n_estimators and other parameters
model.fit(X_train, y_train)

# Predict on the training and testing sets
train_preds = model.predict(X_train)
test_preds = model.predict(X_test)

# Calculate accuracy using accuracy_score
RFtrain_accuracy = accuracy_score(y_train, train_preds)
RFtest_accuracy = accuracy_score(y_test, test_preds)
print("Random Forest Model Scores: \n")
print(f"Train Accuracy: {RFtrain_accuracy:.4f}")
print(f"Test Accuracy: {RFtest_accuracy:.4f}\n")

# K-Fold Cross-Validation (classification accuracy)
kf = KFold(n_splits=10, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')
print("Cross Validation Scores: \n")
print("Cross-validation scores:", scores)
print("Mean accuracy:\n", np.mean(scores), "\n")
RFCrossVal = np.mean(scores)

#%%
print("-----------------------------------------------------------------------\n")
#%%

print("These are the results for the Shift Cipher and Neural Network \n")

#%%

# Use MLPClassifier (Neural Network model)
model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=600, random_state=42)  

model.fit(X_train, y_train)

# Predict on the training and testing sets
train_preds = model.predict(X_train)
test_preds = model.predict(X_test)

# Calculate accuracy using accuracy_score
NNtrain_accuracy = accuracy_score(y_train, train_preds)
NNtest_accuracy = accuracy_score(y_test, test_preds)
print("Neural Network Model Scores: \n")
print(f"Train Accuracy: {NNtrain_accuracy:.4f}")
print(f"Test Accuracy: {NNtest_accuracy:.4f}\n")

# K-Fold Cross-Validation (classification accuracy)
kf = KFold(n_splits=10, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')
print("Cross Validation Scores: \n")
print("Cross-validation scores:", scores)
print("Mean accuracy:", np.mean(scores), "\n")

NNCrossVal = np.mean(scores)

#$$ 

# Results 

print("-----------------------------------------------------------------------\n")

data = {
    "Model": ["Linear Regression", "Logistic Regression", "Decision Tree", "Random Forest", "Neural Network"],
    "Train Score": [train_r2, LLRtrain_accuracy, DTtrain_accuracy, RFtrain_accuracy, NNtrain_accuracy],  
    "Test Score": [test_r2, LLRtest_accuracy, DTtest_accuracy, RFtest_accuracy, NNtest_accuracy],
    "Cross-Validation Score": [LRCrossVal, LLRCrossVal, DTCrossVal, RFCrossVal, NNCrossVal]
}

df_performance = pd.DataFrame(data).fillna("-")  # Replacing None with '-'

print("Shift Cipher Machine Learning Results")
print(df_performance.round(4))

#%%






