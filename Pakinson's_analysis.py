import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader,TensorDataset ,Dataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
import librosa  # For extracting features from the audio recording
import gradio as gr
import pandas as pd


# Load the dataset
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data'
data = pd.read_csv(url)
# This dataset will be used for training the model for analysis. 

# Name is not required at all
# Status will be our label or target with 1 as parkinson's and 0 as healthy .
features=data.drop(columns=['name','status'],axis=1).values
target=data['status'].values

#Train test split 
train_features,test_features,train_target,test_target=train_test_split(features,target,test_size=0.2,random_state=42)


#Normalising the deata
scaler=StandardScaler()
train_features=scaler.fit_transform(train_features)
test_features=scaler.transform(test_features)

#Using Random Forest Classifier because the data is labeled 
model=RandomForestClassifier()
model.fit(train_features,train_target)
#model.predict(test_features)


# Define global variables
fs = 22050  # Sample rate: Number of audio samples per second
columns = ['zero_crossings', 'spectral_centroid', 'spectral_bandwidth', 'spectral_rolloff', 'rms', 'mfcc1']
sentences = [
    "The quick brown fox jumps over the lazy dog.",
    "She sells seashells by the seashore.",
    "How much wood would a woodchuck chuck if a woodchuck could chuck wood?",
    "Peter Piper picked a peck of pickled peppers.",
    "A skunk sat on a stump and thunk the stump stunk, but the stump thunk the skunk stunk."
]

# Initialize an empty DataFrame to store features
df = pd.DataFrame(columns=columns)

def extract_features(y, sr):
    # Extracting features similar to the Parkinson's dataset to ensure we are using only those features which are required for diagnosis
    zero_crossings = librosa.zero_crossings(y).sum()
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr).mean()
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr).mean()
    rms = librosa.feature.rms(y=y).mean()
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=1).mean()  # Use the first MFCC only

    # Aggregate features
    features = np.array([zero_crossings, spectral_centroid, spectral_bandwidth, spectral_rolloff, rms, mfccs])
    return features

# Gradio function to record audio and extract features
def record_and_extract(audio, sentence_index):
    global df

    # Load the audio file
    y, sr = librosa.load(audio, sr=fs)

    # Extract features
    features = extract_features(y, sr)

    # Save features to DataFrame
    df.loc[sentence_index] = features

    if sentence_index < len(sentences) - 1:
        next_sentence = sentence_index + 1
        return f"Recording complete for sentence {sentence_index + 1}. Please read the next sentence.", next_sentence
    else:
        df.to_csv('audio_features.csv', index=False)
        return "All recordings complete. Features saved to audio_features.csv.", None

# Gradio interface
def interface_step(sentence_index):
    if sentence_index is not None:
        return gr.update(value=f"Please read the following sentence and press Submit when ready:\n{sentences[sentence_index]}", label="Instructions"), sentence_index
    else:
        return gr.update(value="All recordings complete. Features saved to audio_features.csv.", interactive=False, label="Instructions"), None

with gr.Blocks() as interface:
    sentence_index = gr.State(value=0)
    instructions = gr.Textbox(value="Please read the following sentence and press Submit when ready:\nThe quick brown fox jumps over the lazy dog.", label="Instructions", interactive=False)

    audio = gr.Audio(type="filepath", label="Record Audio")
    submit = gr.Button("Submit")

    submit.click(record_and_extract, inputs=[audio, sentence_index], outputs=[instructions, sentence_index]).then(interface_step, inputs=sentence_index, outputs=[instructions, sentence_index])

interface.launch()
