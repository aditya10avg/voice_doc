# **VoiceDoc: Parkinson's Disease Detection Using Audio Features**

**VoiceDoc** is a machine learning model designed to predict whether a person exhibits symptoms of Parkinson's disease by analyzing key features in their voice, such as pitch, clarity, tone, and jitter.

---

## **Getting Started**

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/AdityaAVG/voice_doc.git
    ```
2. **Navigate to the Project Directory**:
    ```bash
    cd voice_doc
    ```
3. **Set Up a Virtual Environment** (recommended to avoid dependency issues):
    #### **For Linux and macOS**:
```bash
python3 -m venv venv
source venv/bin/activate
```
### For windows
```bash
venv\Scripts\activate
```
4. **Install the Required Libraries**:
    ```bash
    pip install torch pandas numpy scikit-learn matplotlib librosa gradio

    ```
5. **Run the Application**:

## **Running on Google Colab**

To run this project on Google Colab, follow these steps:

### **Step 1: Open Google Colab**
1. Go to [Google Colab](https://colab.research.google.com/).
2. Sign in with your Google account.

### **Step 2: Clone the GitHub Repository**
In a new Colab notebook, run the following command to clone the repository:

```python
!git clone https://github.com/AdityaAVG/voice_doc.git
```

### **Step 3: Navigate to repository**
```python
import os
os.chdir('/content/voice_doc')
```
#Install dependencies
```bash
!pip install torch pandas numpy scikit-learn matplotlib librosa gradio
```
#Run



---

## **Tech Stack**

- **Python**: Core language for data processing and application building.
- **Librosa**: Used for extracting audio features like MFCC, spectral centroid, and more.
- **Pandas**: Handles data manipulation and feature storage.
- **Scikit-learn**: Used for training the Random Forest classifier and preprocessing.
- **Gradio**: Provides an interactive user interface to record and process audio.
- **PyTorch**: Imported for potential future deep learning models.

---

## **Features**

- **Audio Feature Extraction**: Extracts relevant features from audio files such as zero-crossings, spectral centroid, spectral bandwidth, and MFCC (Mel-frequency cepstral coefficients).
- **Parkinson's Prediction Model**: Utilizes a Random Forest classifier trained on the UCI Parkinson's dataset.
- **Interactive UI**: Gradio-powered interface to record audio, process features, and save them for diagnosis.

---

## **Dataset**

The dataset used for training is the [UCI Parkinson’s dataset](https://archive.ics.uci.edu/ml/datasets/parkinsons), which contains biomedical voice measurements, including features representing fundamental frequencies, harmonics, and noise measurements.

---

## **Gradio Interface**

The Gradio interface allows users to:
1. Record their voice by reading the provided sentences.
2. Extract and analyze relevant voice features.
3. Automatically save extracted features into a CSV file (`audio_features.csv`).

---

## **Usage Instructions**

1. Run the Gradio app with:
    ```bash
    python app.py
    ```

2. Follow the prompts in the UI:
   - **Read** the displayed sentence and **submit** the recording.
   - Continue for all sentences. After completion, the features will be saved in a CSV file.

---

## **Future Improvements**

- **Deep Learning**: Integration of PyTorch for a more sophisticated neural network model.
- **Real-Time Detection**: Enable real-time classification and feedback for audio recordings.
- **Performance Metrics**: Add more comprehensive evaluation metrics like accuracy, precision, recall, and F1 score.
- **User Feedback**: Gather user feedback to improve the prediction model and interface usability.

---

## **References**

- **UCI Parkinson's Dataset**: [Link to Dataset](https://archive.ics.uci.edu/ml/datasets/parkinsons)

---

## **Contributing**

Contributions are welcome! Please fork this repository and submit a pull request if you'd like to improve the project.

---

## **License**

This project is licensed under the MIT License.
