from tqdm import tqdm
import torch
from dataloader import train_dl,val_dl
from model import AudioClassifier
import torch.nn as nn


model=AudioClassifier()
device="cuda" if torch.cuda.is_available() else "cpu"

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001,
                                            steps_per_epoch=int(len(train_dl)),
                                            epochs=20,
                                            anneal_strategy='linear')



def train_one_epoch(model, dataloader, criterion, optimizer, device, evaluate_interval=50):
    model.train()
    total_loss = 0.0
    total_correct = 0
    batch_count = 0

    progress_bar = tqdm(dataloader, desc="Training", unit="batch")

    for inputs, labels in progress_bar:
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Compute accuracy
        _, predicted = torch.max(outputs, 1)
        total_correct += torch.sum(predicted == labels).item()

        # Accumulate loss
        total_loss += loss.item() * inputs.size(0)

        # Update progress bar
        progress_bar.set_postfix({"Loss": loss.item()})

    # Compute epoch metrics
    epoch_loss = total_loss / len(dataloader.dataset)
    epoch_acc = total_correct / len(dataloader.dataset)

    return epoch_loss, epoch_acc

# Function to train the model
def train_model(model, train_dataloader, val_dataloader, criterion, optimizer, num_epochs, device):
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        # Train one epoch
        train_loss, train_acc = train_one_epoch(model, train_dataloader, criterion, optimizer, device)

        # Log training metrics using WandB
        #wandb.log({"epoch": epoch+1, "train_loss": train_loss, "train_acc": train_acc})

        # Evaluate on validation set
        val_loss, val_acc = evaluate_model(model, val_dataloader, criterion, device)

        # Log validation metrics using WandB
        #wandb.log({"epoch": epoch+1, "val_loss": val_loss, "val_acc": val_acc})
        
        # Print epoch results
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"  Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")

        # Save the best model based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
        
    # End the WandB logging session
    #wandb.finish()

# Function to evaluate the model
def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0

    progress_bar = tqdm(dataloader, desc="Evaluation", unit="batch")

    with torch.no_grad():
        for inputs, labels in progress_bar:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs, 1)
            total_correct += torch.sum(predicted == labels).item()
            total_loss += loss.item() * inputs.size(0)

            progress_bar.set_postfix({"Loss": loss.item()})

    epoch_loss = total_loss / len(dataloader.dataset)
    epoch_acc = total_correct / len(dataloader.dataset)

    return epoch_loss, epoch_acc


if __name__=="__main__":
    train_model(model, train_dl, val_dl, criterion, optimizer, num_epochs=20, device=device)