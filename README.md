# Parkinson-s-Voice-analysis-for-Early-Detection
This is the code for model that tries to early detect Parkinson's Symptoms using your vocal data that is your voice pitch , clarity,  tone and jitters  in the voice 

Required Libraries are 
torch
pandas
numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader,TensorDataset ,Dataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
torch.nn
torch.optim
matplotlib.pyplot
pandas 
librosa  # For extracting features from the audio recording
gradio
pandas
