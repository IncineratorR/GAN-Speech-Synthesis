import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Define the GAN-TTS model evaluation function
def evaluate_gan_tts(gan_tts_model, dataloader, device):
    gan_tts_model.eval()
    criterion_mse = nn.MSELoss()
    criterion_mae = nn.L1Loss()

    total_mse_loss = 0.0
    total_mae_loss = 0.0

    with torch.no_grad():
        for mel_batch, text_emb_batch in dataloader:
            mel_batch, text_emb_batch = mel_batch.to(device), text_emb_batch.to(device)

            # Generator forward pass
            generated_audio, _, _, _, _ = gan_tts_model(mel_batch, text_emb_batch)

            # Calculate MSE and MAE loss for mel-spectrogram reconstruction
            mse_loss = criterion_mse(generated_audio, mel_batch)
            mae_loss = criterion_mae(generated_audio, mel_batch)

            total_mse_loss += mse_loss.item()
            total_mae_loss += mae_loss.item()

    # Calculate average MSE and MAE loss
    avg_mse_loss = total_mse_loss / len(dataloader)
    avg_mae_loss = total_mae_loss / len(dataloader)

    return avg_mse_loss, avg_mae_loss

# Assuming you have already trained the GAN-TTS model and have a test dataloader 'test_dataloader'
# loaded with test data, you can evaluate the model as follows:

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Instantiate the GAN-TTS model
mel_input_dim = 80
text_embedding_dim = 768
gen_hidden_dim = 256
gen_output_dim = mel_input_dim
disc_hidden_dim = 256
disc_output_dim = 1

gan_tts_model = GAN_TTS(mel_input_dim, text_embedding_dim, gen_hidden_dim, gen_output_dim, disc_hidden_dim, disc_output_dim)
gan_tts_model.to(device)

# Load pre-trained model weights (if available)
checkpoint_path = "gan_tts_best_model.pt"
if os.path.exists(checkpoint_path):
    gan_tts_model.load_state_dict(torch.load(checkpoint_path))
    print("Pre-trained model weights loaded successfully.")

# Create the test dataset and data loader with the custom collate function
test_dataset = GAN_TTS_Dataset(dataset_dir, max_mel_length)
test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=custom_collate_fn)

# Evaluate the GAN-TTS model
avg_mse_loss, avg_mae_loss = evaluate_gan_tts(gan_tts_model, test_dataloader, device)

print(f"Average MSE Loss: {avg_mse_loss:.4f}")
print(f"Average MAE Loss: {avg_mae_loss:.4f}")
