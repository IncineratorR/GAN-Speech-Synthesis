import os
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from tqdm import tqdm

# Define the GAN-TTS model
class GAN_TTS(nn.Module):
    def __init__(self, mel_input_dim, text_embedding_dim, gen_hidden_dim, gen_output_dim, disc_hidden_dim, disc_output_dim):
        super(GAN_TTS, self).__init__()
        
        # Feed-forward generator
        self.generator = Generator(mel_input_dim, text_embedding_dim, gen_hidden_dim, gen_output_dim)
        
        # Discriminators
        self.discriminator_realism = Discriminator(mel_input_dim)
        self.discriminator_utterance = Discriminator(text_embedding_dim)
        
    def forward(self, mel_batch, text_emb_batch):
        # Generator forward pass
        generated_audio = self.generator(text_emb_batch)
        
        # Discriminator forward pass
        real_realism = self.discriminator_realism(mel_batch)
        fake_realism = self.discriminator_realism(generated_audio.detach())
        
        real_utterance = self.discriminator_utterance(text_emb_batch)
        fake_utterance = self.discriminator_utterance(text_emb_batch)
        
        return generated_audio, real_realism, fake_realism, real_utterance, fake_utterance

# Define the generator
class Generator(nn.Module):
    def __init__(self, mel_input_dim, text_embedding_dim, hidden_dim, output_dim):
        super(Generator, self).__init__()
        
        # Define layers for the generator
        self.fc1 = nn.Linear(text_embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, text_emb_batch):
        # Generator forward pass
        x = F.relu(self.fc1(text_emb_batch))
        x = F.relu(self.fc2(x))
        generated_audio = torch.tanh(self.fc3(x))
        return generated_audio

# Define the discriminator
class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        
        # Define layers for the discriminator
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)
        
    def forward(self, input_data):
        # Discriminator forward pass
        x = F.relu(self.fc1(input_data))
        x = F.relu(self.fc2(x))
        validity = torch.sigmoid(self.fc3(x))
        return validity

# Define DataLoader
class GAN_TTS_Dataset(Dataset):
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

# Instantiate the GAN-TTS model
mel_input_dim = 80  # Mel-spectrogram input dimension
text_embedding_dim = 768  # Text embedding dimension (BERT hidden size)
gen_hidden_dim = 256  # Generator hidden dimension
gen_output_dim = mel_input_dim  # Generator output dimension (mel-spectrogram dimension)
disc_hidden_dim = 256  # Discriminator hidden dimension
disc_output_dim = 1  # Discriminator output dimension (single value for binary classification)

gan_tts_model = GAN_TTS(mel_input_dim, text_embedding_dim, gen_hidden_dim, gen_output_dim, disc_hidden_dim, disc_output_dim)

# Set hyperparameters for training
num_epochs = 10
batch_size = 8
learning_rate = 1e-4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create the dataset and data loader with the custom collate function
dataset = GAN_TTS_Dataset(dataset_dir, max_mel_length)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)

# Function to train the GAN-TTS model
def train_gan_tts(gan_tts_model, dataloader, num_epochs, learning_rate, device):
    gan_tts_model.to(device)
    criterion_realism = nn.BCELoss()
    criterion_utterance = nn.MSELoss()
    optimizer_gen = optim.Adam(gan_tts_model.generator.parameters(), lr=learning_rate)
    optimizer_disc_realism = optim.Adam(gan_tts_model.discriminator_realism.parameters(), lr=learning_rate)
    optimizer_disc_utterance = optim.Adam(gan_tts_model.discriminator_utterance.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        gan_tts_model.train()
        total_gen_loss = 0.0
        total_disc_realism_loss = 0.0
        total_disc_utterance_loss = 0.0

        for mel_batch, text_emb_batch, mel_lengths in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False):
            mel_batch, text_emb_batch = mel_batch.to(device), text_emb_batch.to(device)

            # Generator forward pass
            generated_audio, real_realism, fake_realism, real_utterance, fake_utterance = gan_tts_model(mel_batch, text_emb_batch)

            # ... (Rest of the training loop remains the same)

        # Calculate average loss for this epoch
        avg_gen_loss = total_gen_loss / len(dataloader)
        avg_disc_realism_loss = total_disc_realism_loss / len(dataloader)
        avg_disc_utterance_loss = total_disc_utterance_loss / len(dataloader)

        # Print the average losses for this epoch
        print(f"Epoch {epoch + 1}/{num_epochs}: Gen Loss: {avg_gen_loss:.4f}, Disc Realism Loss: {avg_disc_realism_loss:.4f}, Disc Utterance Loss: {avg_disc_utterance_loss:.4f}")

        # Save model checkpoints after each epoch
        checkpoint_path = f"gan_tts_epoch_{epoch + 1}.pt"
        torch.save(gan_tts_model.state_dict(), checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path}")

# Train the GAN-TTS model
train_gan_tts(gan_tts_model, dataloader, num_epochs, learning_rate, device)
