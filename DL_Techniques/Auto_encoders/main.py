import torch
from torchvision import datasets, transforms
import architecture


# Hyperparameters
batch_size = 32
learning_rate = 1e-3
epochs = 30
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset = datasets.MNIST(root="data", train=True, download=True, transform=transforms.ToTensor())
test_dataset = datasets.MNIST(root="data", train=False, transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

model = architecture.Autoencoder()
loss_function = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-8)
model.to(device)


for epoch in range(epochs):
    losses = []
    for images, _ in train_loader:
        images = images.to(device)
        reconstructed_image = model(images)
        loss = loss_function(reconstructed_image, images)
        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch: {epoch+1}/{epochs}, Train loss: {sum(losses)/len(losses):.4f}")
    losses = []
    for images, _ in test_loader:
        images = images.to(device)
        reconstructed_image = model(images)
        loss = loss_function(reconstructed_image, images)
        losses.append(loss.item())
    print(f"Epoch: {epoch+1}/{epochs}, Test loss: {sum(losses)/len(losses):.4f}")


model_dict = {
    "epochs": epochs,
    "learning rate": learning_rate,
    "batch_size": batch_size,
    "epochs": epochs,
    "loss": "MSEloss",
    "model_state": model.state_dict(),
    "optimizer_state": optimizer.state_dict()
}
torch.save(model_dict, "Trained_Convo_Autoencoder.pth")
