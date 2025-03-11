import random
from collections import Counter
import pandas as pd
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None)
import os
import nltk
nltk.download('punkt')
os.chdir(r'C:\Users\Klaws\Downloads\ML_Crypto Research')
import re
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd
from sklearn.preprocessing import StandardScaler
import itertools
from nltk.util import ngrams
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# %% custom functions

# text cleaning
def remove_non_letters(text):
    return re.sub(r'[^A-Za-z\s]', '', text)  # Only keep letters and spaces


# encryption via affine cipher
def affine_cipher(plaintext, a, b):
    ciphertext = ""
    plaintext = remove_non_letters(plaintext)
    plaintext = plaintext.replace(" ", "")
    plaintext = plaintext.lower()

    for char in plaintext:
        # Apply affine cipher formula: E(x) = (a * x + b) mod 26
        if 'a' <= char <= 'z':
            x = ord(char) - ord('a')
            ciphertext += chr(((a * x + b) % 26) + ord('a'))
        
    ciphertext = ciphertext.upper()
    return plaintext, ciphertext


# decryption via affine cipher
import pandas as pd

# Assuming the decrypt_with_affine function and score_text function are defined
def decrypt_affine(ciphertext, a, b):
    # Decrypt using affine cipher: x = a_inv * (y - b) % 26
    a_inv = mod_inverse(a, 26)
    if a_inv is None:
        return ""  # Return an empty string if a_inv is None (invalid key)
    
    plaintext = ""
    for char in ciphertext:
        if char.isalpha():
            x = (a_inv * (ord(char.upper()) - 65 - b)) % 26 + 65
            plaintext += chr(x)
        else:
            plaintext += char
    return plaintext



def mod_inverse(a, m):
    # Find the modular inverse of a under modulo m
    for i in range(1, m):
        if (a * i) % m == 1:
            return i
    return None  # If no modular inverse exists (e.g., when a is not coprime with m)

def crack_affine_cipher_auto(ciphertext):
    # Try all possible values of a (1-25) and b (0-25) and score the resulting plaintexts
    best_a = 0
    best_b = 0
    best_score = -1
    best_plaintext = ""
    
    for a in range(1, 26):  # a must be coprime with 26 (i.e., GCD(a, 26) = 1)
        if mod_inverse(a, 26) is None:
            continue  # Skip a if it does not have a modular inverse
        
        for b in range(26):
            decrypted = decrypt_affine(ciphertext, a, b)
            if decrypted == "":  # Skip if decryption failed (a_inv was None)
                continue
            
            current_score = score_text(decrypted)
            if current_score > best_score:
                best_score = current_score
                best_a = a
                best_b = b
                best_plaintext = decrypted
    
    return best_a, best_b, best_plaintext



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
    score = 0
    for letter, count in text_counter.items():
        if letter in english_letter_freq:
            score += (count / text_len) * english_letter_freq[letter]
    return score

def bigram_frequency(text):
    # Remove non-alphabetic characters and convert to uppercase to avoid case mismatch
    text = re.sub(r'[^A-Za-z]', '', text).upper()
    
    # If text is too short to form bigrams, return a dictionary with zeros for all bigrams
    if len(text) < 2:
        return {f'Bigram_{i}': 0 for i in range(500)}  # Return a row of zeros
    
    # Create a list of bigrams (pairs of consecutive characters)
    bigrams = [text[i:i+2] for i in range(len(text) - 1)]
    
    # Count the occurrences of each bigram
    bigram_counts = Counter(bigrams)
    
    # Initialize a dictionary with all bigrams as keys and 0 as default count
    bigram_dict = {f'Bigram_{i}': 0 for i in range(500)}
    
    # Update bigram counts in the dictionary
    for bigram, count in bigram_counts.items():
        bigram_index = (ord(bigram[0]) - 65) * 26 + (ord(bigram[1]) - 65)
        bigram_dict[f'Bigram_{bigram_index}'] = count
    
    return bigram_dict



def trigram_frequency(text):
    # Remove non-alphabetic characters and convert to uppercase to avoid case mismatch
    text = re.sub(r'[^A-Za-z]', '', text).upper()
    
    # If text is too short to form trigrams, return a dictionary with zeros for all trigrams
    if len(text) < 3:
        return {f'Trigram_{i}': 0 for i in range(500)}  # Return a row of zeros
    
    # Create a list of trigrams (triplets of consecutive characters)
    trigrams = [text[i:i+3] for i in range(len(text) - 2)]
    
    # Count the occurrences of each trigram
    trigram_counts = Counter(trigrams)
    
    # Initialize a dictionary with all trigrams as keys and 0 as default count
    trigram_dict = {f'Trigram_{i}': 0 for i in range(500)}
    
    # Update trigram counts in the dictionary
    for trigram, count in trigram_counts.items():
        trigram_index = (ord(trigram[0]) - 65) * 26 * 26 + (ord(trigram[1]) - 65) * 26 + (ord(trigram[2]) - 65)
        trigram_dict[f'Trigram_{trigram_index}'] = count
    
    return trigram_dict



def index_of_coincidence(text):
    text = re.sub(r'[^A-Za-z]', '', text)
    N = len(text)
    if N <= 1:
        return 0
    freq = Counter(text.upper())
    ic = sum(f * (f - 1) for f in freq.values()) / (N * (N - 1))
    return ic

# %% Text Preprocessing

# Split the text into chunks of words
def split_text_into_chunks_by_words(text, chunk_size=10):
    words = text.split()  # Split the text into individual words
    chunks = []
    current_chunk = []

    for word in words:
        # Add words to the current chunk until the chunk reaches the desired size
        current_chunk.append(word)
        if len(current_chunk) >= chunk_size:
            chunks.append(' '.join(current_chunk))
            current_chunk = []  # Start a new chunk

    # Add any remaining words as the final chunk
    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks


# %% Read in text file
with open('StarWars1.txt', 'r', encoding='utf-8') as file:
    chapter_text = file.read()

chunks = (split_text_into_chunks_by_words(chapter_text, chunk_size=10))
# %% Encrypt and store sentences

plain_cipher_pairs = []
for chunk in chunks:
    a = random.choice([1, 3, 5, 7, 11, 15, 17, 19, 21, 23, 25])  # Random 'a' (must be coprime with 26)
    b = random.randint(0, 25)  # Random 'b' for affine cipher
    plaintext, ciphertext = affine_cipher(chunk, a, b)
    plain_cipher_pairs.append((a, b, plaintext, ciphertext))  
    
    
    # gcd function to compute greatest common divisor
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a

#%%


df = pd.DataFrame(plain_cipher_pairs, columns=['A', 'B', 'Plaintext', 'Ciphertext'])
df = df.iloc[:-1].copy()

# %% Full features for ML model training

df_ml_dataset = df.copy()
df_ml_dataset['Ciphertext_IC'] = df_ml_dataset['Ciphertext'].apply(index_of_coincidence)
char_freqs = df_ml_dataset['Ciphertext'].apply(char_frequency)
char_freqs = char_freqs.fillna(0)
bigram_freqs = df_ml_dataset['Ciphertext'].apply(bigram_frequency)
bigram_freqs = bigram_freqs.fillna(0)
trigram_freqs = df_ml_dataset['Ciphertext'].apply(trigram_frequency)
trigram_freqs = trigram_freqs.fillna(0)

trigram_columns = [f'Trigram_{i}' for i in range(500)]  
trigram_df = pd.DataFrame(trigram_freqs.tolist(), columns=trigram_columns)
char_freq_df = pd.DataFrame(char_freqs.tolist(), columns=[f'Letter_{char}' for char in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'])
bigram_columns = [f'Bigram_{i}' for i in range(500)]  
bigram_df = pd.DataFrame(bigram_freqs.tolist(), columns=bigram_columns)
df_ml_dataset = pd.concat([df_ml_dataset, char_freq_df, bigram_df, trigram_df], axis=1)
df_ml_dataset = df_ml_dataset.fillna(0)



print(f"Generated {len(df)} encrypted chunks with Affine cipher.\n")

# %%

df_cryptanalysis = df.copy()

df_cryptanalysis['best_a'] = None
df_cryptanalysis['best_b'] = None
df_cryptanalysis['best_plaintext'] = None

for i in range(df_cryptanalysis.shape[0]):
    ciphertext = df_cryptanalysis.loc[i, 'Ciphertext']
    best_a, best_b, best_plaintext = crack_affine_cipher_auto(ciphertext)
    
    df_cryptanalysis.loc[i, 'best_a'] = best_a
    df_cryptanalysis.loc[i, 'best_b'] = best_b
    df_cryptanalysis.loc[i, 'best_plaintext'] = best_plaintext
    
df_cryptanalysis['best_a'] = pd.to_numeric(df_cryptanalysis['best_a'], errors='coerce')
df_cryptanalysis['best_b'] = pd.to_numeric(df_cryptanalysis['best_b'], errors='coerce')

# You might want to compare the results to the actual keys in the dataset (if available)
df_cryptanalysis['keys_match'] = (df_cryptanalysis['A'] == df_cryptanalysis['best_a']) & (df_cryptanalysis['B'] == df_cryptanalysis['best_b'])

print("Manual Cryptanalysis Scores: \n")
false_count = len(df_cryptanalysis) - df_cryptanalysis['keys_match'].sum()
print(f"CRYPTANALYSIS: Number of non-matching keys: {false_count}")
acc_perc = (len(df_cryptanalysis) - false_count) / len(df_cryptanalysis)
print(f"CRYPTANALYSIS: Percent of matching keys: {acc_perc:.4f}\n")

#%%

# %% ML model train-test for Combined Affine Cipher

X = df_ml_dataset.drop(columns=['A', 'B', 'Plaintext', 'Ciphertext'])
y = df_ml_dataset[['A', 'B']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#%%

print("-----------------------------------------------------------------------\n")

#%%

print("These are the results for the Affine Cipher and Logistic Regression\n")


# Step 1: Train the model for predicting 'A'
LLR_model_A = LogisticRegression(max_iter=2000)
LLR_model_A.fit(X_train, y_train['A'])

train_accuracy_A = LLR_model_A.score(X_train, y_train['A'])
print(f"a ML Model train accuracy: {train_accuracy_A:.4f}")

accuracy_A = LLR_model_A.score(X_test, y_test['A'])
print(f"a ML Model test accuracy: {accuracy_A:.4f}")

# Step 2: Make predictions for 'A'
train_preds_A = LLR_model_A.predict(X_train)
test_preds_A = LLR_model_A.predict(X_test)

# Step 3: Add the predictions for 'A' as new features to the training and test data
X_train['A_pred'] = train_preds_A
X_test['A_pred'] = test_preds_A

# Step 4: Train the model for predicting 'B' using the new features
LLR_model_B = LogisticRegression(max_iter=2000)
LLR_model_B.fit(X_train, y_train['B'])

LLR_train_accuracy_B = LLR_model_B.score(X_train, y_train['B'])
print(f"b ML Model train accuracy: {LLR_train_accuracy_B:.4f}")

LLR_test_accuracy_B = LLR_model_B.score(X_test, y_test['B'])
print(f"b ML Model test accuracy: {LLR_test_accuracy_B:.4f}")

# Step 5: Cross-validation for model 'B'
cv = StratifiedKFold(n_splits=5)
LLR_Cross_Val = cross_val_score(LLR_model_B, X_train, y_train['B'], cv=cv).mean()
print(f"b ML Model Cross Validation accuracy: {LLR_Cross_Val:.4f}")





#%%

print("-----------------------------------------------------------------------\n")

#%%

print("These are the results for the Affine Cipher and Decision Trees\n")


# Step 1: Train the decision tree model for predicting 'A'
DTC_model_A = DecisionTreeClassifier(
    criterion='gini',  
    max_depth=20,    
    min_samples_leaf=1, 
    min_samples_split=2, 
    random_state=42
)

DTC_model_A.fit(X_train, y_train["A"])

# Train accuracy for 'A'
train_accuracy_A = DTC_model_A.score(X_train, y_train['A'])
print(f"a ML Model train accuracy: {train_accuracy_A:.4f}")

# Test accuracy for 'A'
accuracy_A = DTC_model_A.score(X_test, y_test['A'])
print(f"a ML Model test accuracy: {accuracy_A:.4f}")

# Step 2: Predict 'A'
train_preds_A = DTC_model_A.predict(X_train)
test_preds_A = DTC_model_A.predict(X_test)

# Add the predictions for 'A' as new features to the training and test data
X_train['A_pred'] = train_preds_A
X_test['A_pred'] = test_preds_A

# Step 3: Train the decision tree model for predicting 'B' using the augmented data
DTC_model_B = DecisionTreeClassifier(
    criterion='gini',  
    max_depth=20,    
    min_samples_leaf=1, 
    min_samples_split=2, 
    random_state=42
)

DTC_model_B.fit(X_train, y_train['B'])

# Train accuracy for 'B'
DTC_train_accuracy_B = DTC_model_B.score(X_train, y_train['B'])
print(f"b ML Model train accuracy: {DTC_train_accuracy_B:.4f}")

# Test accuracy for 'B'
DTC_test_accuracy_B = DTC_model_B.score(X_test, y_test['B'])
print(f"b ML Model test accuracy: {DTC_test_accuracy_B:.4f}")

# Step 4: Cross-validation for model 'B'
cv = StratifiedKFold(n_splits=5)
DTC_Cross_Val = cross_val_score(DTC_model_B, X_train, y_train['B'], cv=cv).mean()
print(f"b ML Model Cross Validation accuracy: {DTC_Cross_Val:.4f}")






# %%
print("-----------------------------------------------------------------------\n")

#%%

print("These are the results for the Affine Cipher and Random Forest\n")



RF_model = RandomForestClassifier(
    n_estimators=3000,
    max_depth=None,         
    min_samples_split=10,  
    min_samples_leaf=1,   
    max_features=1, 
    random_state=42,      
    bootstrap=False        
)

# Step 1: Train the model to predict 'A'
RF_model.fit(X_train, y_train['A'])

# Train accuracy for 'A'
train_accuracy_A = RF_model.score(X_train, y_train['A'])
print(f"a ML Model train accuracy: {train_accuracy_A:.4f}")

# Test accuracy for 'A'
accuracy_A = RF_model.score(X_test, y_test['A'])
print(f"a ML Model test accuracy: {accuracy_A:.4f}")

# Step 2: Predict 'A'
train_preds_A = RF_model.predict(X_train)
test_preds_A = RF_model.predict(X_test)

# Add the predictions for 'A' as new features to the training and test data
X_train['A_pred'] = train_preds_A
X_test['A_pred'] = test_preds_A

# Step 3: Train the model to predict 'B' using the augmented data
RF_model.fit(X_train, y_train['B'])

# Train accuracy for 'B'
RF_train_accuracy_B = RF_model.score(X_train, y_train['B'])
print(f"b ML Model train accuracy: {RF_train_accuracy_B:.4f}")

# Test accuracy for 'B'
RF_test_accuracy_B = RF_model.score(X_test, y_test['B'])
print(f"b ML Model test accuracy: {RF_test_accuracy_B:.4f}")

# Step 4: Cross-validation for model 'B'
cv = StratifiedKFold(n_splits=5)
RF_Cross_Val = cross_val_score(RF_model, X_train, y_train['B'], cv=cv).mean()
print(f"b ML Model Cross Validation accuracy: {RF_Cross_Val:.4f}")



# %%

#%%
print("-----------------------------------------------------------------------\n")

#%%

print("These are the results for the Affine Cipher and Neural Network\n")


NN_model = MLPClassifier(
    hidden_layer_sizes=(3000), 
    activation='relu', 
    solver='adam', 
    alpha=0.0001,
    learning_rate_init=0.0005, 
    max_iter=3000,
    verbose=0,  
    random_state=42
)

# Step 1: Train the model to predict 'A'
NN_model.fit(X_train, y_train['A'])

# Train accuracy for 'A'
train_accuracy_A = NN_model.score(X_train, y_train['A'])
print(f"a ML Model train accuracy: {train_accuracy_A:.4f}")

# Test accuracy for 'A'
accuracy_A = NN_model.score(X_test, y_test['A'])
print(f"a ML Model test accuracy: {accuracy_A:.4f}")

# Step 2: Predict 'A'
train_preds_A = NN_model.predict(X_train)
test_preds_A = NN_model.predict(X_test)

# Add the predictions for 'A' as new features to the training and test data
X_train['A_pred'] = train_preds_A
X_test['A_pred'] = test_preds_A

# Step 3: Train the model to predict 'B' using the augmented data
NN_model.fit(X_train, y_train['B'])

# Train accuracy for 'B'
NN_train_accuracy_B = NN_model.score(X_train, y_train['B'])
print(f"b ML Model train accuracy: {NN_train_accuracy_B:.4f}")

# Test accuracy for 'B'
NN_test_accuracy_B = NN_model.score(X_test, y_test['B'])
print(f"b ML Model test accuracy: {NN_test_accuracy_B:.4f}")

# Step 4: Cross-validation for model 'B'
cv = StratifiedKFold(n_splits=5)
NN_Cross_Val = cross_val_score(NN_model, X_train, y_train['B'], cv=cv).mean()
print(f"b ML Model Cross Validation accuracy: {NN_Cross_Val:.4f}")



#%%
print("-----------------------------------------------------------------------\n")

#%%

print("These are the results for the Affine Cipher and Gradient Boosting\n")


GB_model = GradientBoostingClassifier(
    n_estimators=500,        
    learning_rate=0.05,      
    max_depth=5,             
    min_samples_split=30,     
    min_samples_leaf=5,     
    subsample=0.9,           
    max_features=2,     
    random_state=42          
)
''
# Step 1: Train the GB model to predict 'A'
GB_model.fit(X_train, y_train['A'])

# Train accuracy for 'A'
train_accuracy_A = GB_model.score(X_train, y_train['A'])
print(f"a ML Model train accuracy: {train_accuracy_A:.4f}")

# Test accuracy for 'A'
accuracy_A = GB_model.score(X_test, y_test['A'])
print(f"a ML Model test accuracy: {accuracy_A:.4f}")

# Step 2: Predict 'A'
train_preds_A = GB_model.predict(X_train)
test_preds_A = GB_model.predict(X_test)

# Add the predictions for 'A' as new features to the training and test data
X_train['A_pred'] = train_preds_A
X_test['A_pred'] = test_preds_A

# Step 3: Train the model to predict 'B' using the augmented data
GB_model.fit(X_train, y_train['B'])

# Train accuracy for 'B'
GB_train_accuracy_B = GB_model.score(X_train, y_train['B'])
print(f"b ML Model train accuracy: {GB_train_accuracy_B:.4f}")

# Test accuracy for 'B'
GB_test_accuracy_B = GB_model.score(X_test, y_test['B'])
print(f"b ML Model test accuracy: {GB_test_accuracy_B:.4f}")

# Step 4: Cross-validation for model 'B'
cv = StratifiedKFold(n_splits=5)
GB_Cross_Val = cross_val_score(GB_model, X_train, y_train['B'], cv=cv).mean()
print(f"b ML Model Cross Validation accuracy: {GB_Cross_Val:.4f}")





#%%

print("-----------------------------------------------------------------------\n")

# Complete the data dictionary with the results
data = {
    "Model": ["Logistic Regression", "Decision Tree", "Random Forest", "Neural Network", "Gradient Boosting", "Manual Cryptanalysis"],
    "Overall Train Accuracy": [LLR_train_accuracy_B, DTC_train_accuracy_B, RF_train_accuracy_B, NN_train_accuracy_B, GB_train_accuracy_B, "-"],
    "Overall Test Accuracy": [LLR_test_accuracy_B, DTC_test_accuracy_B, RF_test_accuracy_B, NN_test_accuracy_B, GB_test_accuracy_B, "-"],
    "Cross-Val Accuracy": [LLR_Cross_Val, DTC_Cross_Val, RF_Cross_Val, NN_Cross_Val, GB_Cross_Val, "-"],
    "Keys Match (Manual)": ["-", "-", "-", "-", "-", acc_perc]
}

# Create a DataFrame to display the comparison
model_comparison_df = pd.DataFrame(data)
print(model_comparison_df)

