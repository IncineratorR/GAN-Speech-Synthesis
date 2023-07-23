import os
import numpy as np
import librosa
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel

# Define the path to the LJSpeech dataset directory
dataset_dir = '/content/drive/MyDrive/dataset/LJSpeech-1.1'

# Define hyperparameters for audio processing
sampling_rate = 22050  # Sampling rate of the audio
n_fft = 1024  # Size of the FFT window
hop_length = 256  # Number of samples between successive frames

# Load a pre-trained BERT tokenizer and model for text embedding
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")

# Define the maximum length of the mel-spectrograms (adjust as needed)
max_mel_length = 770

# Function to preprocess text and convert it to text embeddings
def preprocess_text(text):
    # Tokenize the text and convert it to numerical IDs with attention mask
    inputs = tokenizer.encode_plus(text, add_special_tokens=True, truncation=True, max_length=128, padding='max_length', return_tensors='pt')
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    # Get the BERT model's hidden states for the input text
    with torch.no_grad():
        hidden_states = bert_model(input_ids, attention_mask=attention_mask)[0]

    # Get the last hidden state of the [CLS] token as the text embedding
    text_embedding = hidden_states[:, 0, :].squeeze()
    return text_embedding

# Function to load and preprocess audio files
def preprocess_audio(audio_file_path):
    # Load audio and scale it to the range [-1, 1]
    waveform, _ = librosa.load(audio_file_path, sr=sampling_rate)
    waveform = torch.FloatTensor(waveform)

    # Rescale waveform to the range [-1, 1]
    waveform /= waveform.abs().max()

    mel_spectrogram = librosa.feature.melspectrogram(
        y=waveform.numpy(),  # Convert to numpy array for librosa
        sr=sampling_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=80,  # Number of mel bands
        fmin=0,    # Minimum frequency in mel filter
        fmax=8000  # Maximum frequency in mel filter
    )
    mel_spectrogram = np.log(mel_spectrogram + 1e-9)  # Log-scale mel spectrogram
    return torch.FloatTensor(mel_spectrogram.T)  # Transpose and convert to PyTorch tensor

# Define a custom dataset class to load LJSpeech data
class LJSpeechDataset(Dataset):
    def __init__(self, dataset_dir, max_mel_length):
        self.audio_file_paths = [os.path.join(dataset_dir, 'wavs', filename) for filename in os.listdir(os.path.join(dataset_dir, 'wavs'))]
        self.texts = [line.strip().split('|')[2] for line in open(os.path.join(dataset_dir, 'metadata.csv'), 'r').readlines()]
        self.max_mel_length = max_mel_length

    def __len__(self):
        return len(self.audio_file_paths)

    def __getitem__(self, idx):
        audio_file_path = self.audio_file_paths[idx]
        mel_spectrogram = preprocess_audio(audio_file_path)
        text_embedding = preprocess_text(self.texts[idx])
        return mel_spectrogram, text_embedding

# Custom collate function to handle variable-length data in the batch
def custom_collate_fn(batch):
    # Sort the batch in descending order of mel-spectrogram lengths
    sorted_batch = sorted(batch, key=lambda x: x[0].shape[0], reverse=True)

    # Pad mel-spectrograms to the maximum length in the batch
    max_mel_length = max(sorted_batch, key=lambda x: x[0].shape[0])[0].shape[0]
    padded_mels = [torch.nn.functional.pad(mel[:max_mel_length], (0, 0, 0, max_mel_length - mel.shape[0])) for mel, _ in sorted_batch]
    padded_mels = torch.stack(padded_mels)

    # Prepare the text embeddings
    text_embeddings = torch.stack([text_emb for _, text_emb in sorted_batch])

    # Get the mel-spectrogram lengths for later use
    mel_lengths = [mel.shape[0] for mel, _ in sorted_batch]

    return padded_mels, text_embeddings, mel_lengths

# Create the dataset and data loader with the custom collate function
dataset = LJSpeechDataset(dataset_dir, max_mel_length)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=custom_collate_fn)

# Iterate over the dataset for a few batches and print some information
for mel_batch, text_emb_batch, mel_lengths in dataloader:
    print(f"Mel-spectrogram batch shape: {mel_batch.shape}")
    print(f"Text embedding batch shape: {text_emb_batch.shape}")
    print(f"Mel-spectrogram lengths: {mel_lengths}")
    break
