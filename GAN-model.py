import torch
import torch.nn as nn
import torch.nn.functional as F

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

# Instantiate the GAN-TTS model
mel_input_dim = 80  # Mel-spectrogram input dimension
text_embedding_dim = 768  # Text embedding dimension (BERT hidden size)
gen_hidden_dim = 256  # Generator hidden dimension
gen_output_dim = mel_input_dim  # Generator output dimension (mel-spectrogram dimension)
disc_hidden_dim = 256  # Discriminator hidden dimension
disc_output_dim = 1  # Discriminator output dimension (single value for binary classification)

gan_tts_model = GAN_TTS(mel_input_dim, text_embedding_dim, gen_hidden_dim, gen_output_dim, disc_hidden_dim, disc_output_dim)

# Print the GAN-TTS model architecture
print(gan_tts_model)

