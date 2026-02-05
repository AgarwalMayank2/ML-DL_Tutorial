from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.optim as optim
import torch.nn as nn
import torch
import architecture

# Hyperparameters
batch_size = 32
learning_rate = 1e-3
epochs = 20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset = datasets.MNIST(root="../Auto_encoders/data", train=True, download=True, transform=transforms.ToTensor())
test_dataset = datasets.MNIST(root="../Auto_encoders/data", train=False, transform=transforms.ToTensor())
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

model = architecture.VAE()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-8)
model.to(device)


def vae_loss(x, predicted, mean, log_variance):
    reconstruction_loss = nn.functional.mse_loss(predicted, x)
    kl_loss = -0.5*torch.sum(1+log_variance-mean.pow(2)-log_variance.exp())
    return reconstruction_loss + kl_loss


for epoch in range(epochs):
    print(f"Epoch:- {epoch+1}/{epochs}")
    losses = []
    for x, _ in train_loader:
        x = x.to(device)
        recon_image, mean, log_variance = model(x)
        loss = vae_loss(x, recon_image, mean, log_variance)
        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Training loss:- {sum(losses)/len(losses)}")

    losses = []
    for x, _ in test_loader:
        x = x.to(device)
        recon_image, mean, log_variance = model(x)
        loss = vae_loss(x, recon_image, mean, log_variance)
        losses.append(loss.item())
    print(f"Test loss:- {sum(losses)/len(losses)}")

model_dict = {
    "epochs": epochs,
    "learning rate": learning_rate,
    "batch_size": batch_size,
    "loss": "MSEloss + KL_loss",
    "model_state": model.state_dict(),
    "optimizer_state": optimizer.state_dict()
}
torch.save(model_dict, "Trained_Convolutional_VAE.pth")
