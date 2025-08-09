import torch
import torch.nn as nn
from tqdm import tqdm
from utils.helpers import LOGGER


class SimpleConvAutoencoder(nn.Module):
    """A simple convolutional autoencoder for denoising/reconstructing images."""

    def __init__(self):
        super(SimpleConvAutoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),  # -> 16x16x16
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # -> 32x8x8
            nn.ReLU(),
            nn.Conv2d(32, 64, 7)  # -> 64x2x2
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 7),  # -> 32x8x8
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),  # -> 16x16x16
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1),  # -> 3x32x32
            nn.Sigmoid()  # Bring values to [0, 1] range
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def train_autoencoder(model, dataloader, device, epochs=10, lr=1e-3):
    """Trains the autoencoder on a clean dataset."""
    LOGGER.info("Training a simple convolutional autoencoder for image reconstruction...")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        total_loss = 0
        for data in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}"):
            imgs, _ = data
            imgs = imgs.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, imgs)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        LOGGER.info(f"Epoch {epoch + 1}/{epochs}, Average Reconstruction Loss: {avg_loss:.6f}")
    LOGGER.info("Autoencoder training complete.")
    return model


def reconstruct_images(model, images_tensor, device):
    """Uses a trained autoencoder to reconstruct (and hopefully clean) images."""
    LOGGER.info(f"Reconstructing {len(images_tensor)} images using the autoencoder...")
    model.to(device)
    model.eval()
    with torch.no_grad():
        images_tensor = images_tensor.to(device)
        reconstructed_images = model(images_tensor)
    LOGGER.info("Image reconstruction complete.")
    return reconstructed_images.cpu()