import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision import models
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime


from custom_dataset import CustomDataset

writer = SummaryWriter()

# Split the dataset into train, validation, and test sets
def SplitDataset(dataset):
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    return train_dataset, val_dataset, test_dataset


#save the Model and metrics
def save_model_and_metrics(model, epoch, loss):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_folder = f"model_evaluation/model_{timestamp}"
    weights_folder = os.path.join(model_folder, "weights")
    os.makedirs(weights_folder,exist_ok=True)

    # Save the model
    model_filename = os.path.join(weights_folder,f"model_epoch{epoch}_loss{loss:.4f}.pth")
    torch.save(model.state_dict(), model_filename)

    # save metrics (you can adjust this based on your specific metrics)
    metrics = {"epoch": epoch, "loss": loss}

    metrics_filename = os.path.join(model_folder, "metrics_epoch_{epoch}.json")
    torch.save(metrics, metrics_filename)


# Define the train function
def train(model,dataset,Data,val, epochs=1,unfreeze_layers=2):

    # Define dataloaders for train, validation, and test sets
    train_dataloader = DataLoader(Data, batch_size=200, shuffle=True)
    val_dataloader = DataLoader(val, batch_size=200, shuffle=True)

    # Freeze all layers except the last 'unfreeze_layers' layers
    for param in model.parameters():
        param.requires_grad = False
    for param in model.layer4[-unfreeze_layers:].parameters():
        param.requires_grad = True
    
    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        

        # Training loop
        for i, data in enumerate(train_dataloader, 0):
            inputs, labels_tuple = data
            labels = [dataset.encoder[label] for label in labels_tuple]
            labels = torch.tensor(labels).to(device)
            inputs = inputs.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            # print("Running loss: ", running_loss)

        average_loss = running_loss / len(train_dataloader)
        writer.add_scalar('TRAINING LOSS VS EPOCH', average_loss, epoch+1)
        # Log the learning rate
        for param_group in optimizer.param_groups:
            current_lr = param_group['lr']
            writer.add_scalar('Learning Rate VS EPOCH', current_lr, epoch+1)
        # Calculate and log accuracy
        model.eval()  # Set the model to evaluation mode
        correct = 0
        total = 0
        with torch.no_grad():
            for j, data in enumerate(val_dataloader, 0):
                inputs, labels_tuple = data
                labels = [dataset.encoder[label] for label in labels_tuple]
                labels = torch.tensor(labels).to(device)
                inputs = inputs.to(device)
                
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
       
        accuracy = correct / total
        writer.add_scalar('Validation Accuracy VS EPOCH', accuracy, epoch+1)
        print(f"Epoch {epoch+1}, Average Loss: {average_loss}, Learning Rate: {current_lr}, Accuracy: {accuracy}")
        # save the model and metrics at the end of each epoch
        save_model_and_metrics(model, epoch+1, average_loss)

    print('Finished Training')
    writer.close()



transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = CustomDataset(csv_file='Datasets/training_data.csv', transform=transform)

# Split the dataset into train, validation, and test sets
train_dataset, val_dataset, test_dataset = SplitDataset(dataset)

resnet = models.resnet50(weights='IMAGENET1K_V2')

num_categories = len(dataset.encoder)
num_features = resnet.fc.in_features
resnet.fc = torch.nn.Linear(num_features, num_categories)

# Train the model for a single epoch
train(resnet,dataset,train_dataset,val_dataset, epochs=300,unfreeze_layers=2)


