import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel
import librosa

# Define the path to the trained GAN-TTS model checkpoint
checkpoint_path = "gan_tts_best_model.pt"

# Define hyperparameters for audio processing
sampling_rate = 22050  # Sampling rate of the audio
n_fft = 1024  # Size of the FFT window
hop_length = 256  # Number of samples between successive frames

# Load the pre-trained BERT tokenizer and model for text embedding
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")

# Define a function to preprocess text and convert it to text embeddings
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

# Load the trained GAN-TTS model
class GAN_TTS(nn.Module):
    # ... (Same as the previously defined GAN_TTS class) ...

gan_tts_model = GAN_TTS(mel_input_dim, text_embedding_dim, gen_hidden_dim, gen_output_dim, disc_hidden_dim, disc_output_dim)
gan_tts_model.load_state_dict(torch.load(checkpoint_path))
gan_tts_model.eval()

# Define a function to synthesize speech from input text
def synthesize_speech(input_text, device='cuda'):
    # Preprocess the input text and convert it to text embeddings
    text_embedding = preprocess_text(input_text)
    text_embedding = text_embedding.to(device)

    # Generate speech using the GAN-TTS model
    with torch.no_grad():
        generated_audio, _, _, _, _ = gan_tts_model(None, text_embedding)

    # Convert generated audio to waveform and scale it to the range [-1, 1]
    generated_waveform = generated_audio.squeeze().cpu().numpy()
    generated_waveform = librosa.resample(generated_waveform, gan_tts_model.sampling_rate, sampling_rate)
    generated_waveform /= np.max(np.abs(generated_waveform))

    return generated_waveform

# Example usage:
input_text = "This is an example of text-to-speech synthesis using GAN-TTS."
generated_waveform = synthesize_speech(input_text)
librosa.output.write_wav("generated_speech.wav", generated_waveform, sampling_rate)
