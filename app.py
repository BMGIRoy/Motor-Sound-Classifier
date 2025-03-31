import streamlit as st
import numpy as np
import pandas as pd
from scipy.io import wavfile
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from scipy.fftpack import dct
import os
import zipfile
import tempfile
import shutil

st.set_page_config(page_title="Motor Sound Classifier", layout="centered")

# ------------------ MFCC Feature Extraction ------------------
def extract_mfcc(signal, sample_rate, num_ceps=13, nfft=512):
    emphasized_signal = np.append(signal[0], signal[1:] - 0.97 * signal[:-1])
    frame_size, frame_stride = 0.025, 0.01
    frame_length = int(round(frame_size * sample_rate))
    frame_step = int(round(frame_stride * sample_rate))
    signal_length = len(emphasized_signal)
    num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))
    pad_signal_length = num_frames * frame_step + frame_length
    z = np.zeros((pad_signal_length - signal_length))
    pad_signal = np.append(emphasized_signal, z)
    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + \
              np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]
    frames *= np.hamming(frame_length)
    mag_frames = np.absolute(np.fft.rfft(frames, nfft))
    pow_frames = ((1.0 / nfft) * ((mag_frames) ** 2))
    nfilt = 26
    low_freq_mel = 0
    high_freq_mel = 2595 * np.log10(1 + (sample_rate / 2) / 700)
    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)
    hz_points = 700 * (10**(mel_points / 2595) - 1)
    bin = np.floor((nfft + 1) * hz_points / sample_rate)
    fbank = np.zeros((nfilt, int(np.floor(nfft / 2 + 1))))
    for m in range(1, nfilt + 1):
        f_m_minus, f_m, f_m_plus = int(bin[m - 1]), int(bin[m]), int(bin[m + 1])
        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
    filter_banks = 20 * np.log10(filter_banks)
    mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, :num_ceps]
    return np.mean(mfcc, axis=0)

# ------------------ Model Training Utility ------------------
def train_model_from_data(data):
    X = pd.DataFrame(data['features'])
    y = pd.Series(data['labels'])
    y_encoded = y.map({'normal': 0, 'abnormal': 1})
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=['Normal', 'Abnormal'])
    matrix = confusion_matrix(y_test, y_pred)
    return model, report, matrix

# ------------------ File Processor ------------------
def process_uploaded_zip(zip_file):
    data = {'features': [], 'labels': []}
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(tmpdir)
        for label in ['normal', 'abnormal']:
            folder_path = os.path.join(tmpdir, label)
            if os.path.exists(folder_path):
                for file in os.listdir(folder_path):
                    if file.endswith(".wav"):
                        try:
                            rate, signal = wavfile.read(os.path.join(folder_path, file))
                            features = extract_mfcc(signal, rate)
                            data['features'].append(features)
                            data['labels'].append(label)
                        except:
                            continue
    return data

# ------------------ Streamlit Interface ------------------
st.title("🔊 Motor Sound Classifier")
tabs = st.tabs(["🔍 Predict", "⚙️ Train Custom Model"])

# Initialize session model
if 'custom_model' not in st.session_state:
    st.session_state['custom_model'] = train_model_from_data({
        'features': [[-0.72]*13, [0.17]*13, [3.08]*13, [55.82]*13, [-12.50]*13],
        'labels': ['normal', 'normal', 'abnormal', 'abnormal', 'abnormal']
    })[0]

with tabs[0]:
    st.subheader("🔍 Predict Normal or Abnormal")
    uploaded_file = st.file_uploader("Upload a .wav file", type="wav")
    if uploaded_file is not None:
        try:
            rate, signal = wavfile.read(uploaded_file)
            features = extract_mfcc(signal, rate)
            prediction = st.session_state['custom_model'].predict([features])[0]
            label = "Normal" if prediction == 0 else "Abnormal"
            st.success(f"Prediction: **{label}**")
        except Exception as e:
            st.error(f"Error: {e}")

with tabs[1]:
    st.subheader("⚙️ Train Your Own Model")
    st.write("Upload a ZIP file with folders `/normal/` and `/abnormal/` each containing `.wav` files.")
    training_zip = st.file_uploader("Upload Training ZIP", type="zip", key="zip")
    if training_zip:
        data = process_uploaded_zip(training_zip)
        if len(data['features']) > 4:
            model, report, matrix = train_model_from_data(data)
            st.session_state['custom_model'] = model
            st.success("✅ Model trained and ready for predictions!")
            st.text("Classification Report:")
            st.code(report)
            st.text("Confusion Matrix:")
            st.write(pd.DataFrame(matrix, index=["Actual Normal", "Actual Abnormal"], columns=["Pred Normal", "Pred Abnormal"]))
        else:
            st.warning("⚠️ Not enough samples to train. Minimum ~5 normal and 5 abnormal sounds recommended.")
