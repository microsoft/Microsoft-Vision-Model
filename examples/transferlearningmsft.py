import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import microsoftvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import time
import copy
import os
os.environ["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"

class CustomFCLayer(nn.Module):
    def __init__(self, model, num_classes):
        super(CustomFCLayer, self).__init__()
        self.model = model
        self.fc = nn.Linear(2048, 1024)
        self.relu = nn.ReLU()
        self.dropout_layer = nn.Dropout(p=0.5)
        self.out = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.model(x)  # Run through existing model first
        x = self.dropout_layer(self.relu(self.fc(x)))  # Pass through new layer
        x = self.out(x)
        return x

def main():

    transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize(299),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])])
    data_transforms = {
        'train': transform,
        'val': transform
    }

    data_dir = 'C:\Temp\TorchSigs299'
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x)
                                            ,data_transforms[x])
                    for x in ['train', 'val']}
    
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=32,
                                                shuffle=True, num_workers=2, pin_memory=True)
                for x in ['train', 'val']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    classes = image_datasets['train'].classes

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #print(classes)

    # Store the metrics to be graphed
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []

    def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
        since = time.time()

        best_model_wts = copy.deepcopy(model.state_dict())
    
        with open('bestaccuracy.txt', 'r') as file:
            # Read the double from the file
            best_acc = float(file.read())
            print(f'Load Best Acc: {best_acc}')

        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print('{} Loss: {:.4f} Acc: {:.4f} LR: {}'.format(
                    phase, epoch_loss, epoch_acc, scheduler.get_last_lr()))
                
                if phase == 'train':
                    scheduler.step()
                    train_losses.append(epoch_loss)
                    train_accs.append(epoch_acc)
                else:
                    val_losses.append(epoch_loss)
                    val_accs.append(epoch_acc)

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(model.state_dict(), 'TorchStateTrainMsft.pth')

                    with open('bestaccuracy.txt', 'w') as file:
                        # Write the double to the file
                        file.write(str(torch.tensor(best_acc, device='cpu').item()))
            print()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights
        model.load_state_dict(best_model_wts)
        return model

    #### ConvNet as fixed feature extractor ####
    # Here, we need to freeze all the network except the final layer.
    # We need to set requires_grad == False to freeze the parameters so that the gradients are not computed in backward()
    model_conv = microsoftvision.models.resnet50(pretrained=True)

    for param in model_conv.parameters():
        param.requires_grad = False

    # Parameters of newly constructed modules have requires_grad=True by default
    model_conv = CustomFCLayer(model_conv, len(classes))   

    #print(model_conv)

    if os.path.exists('TorchStateTrainMsft.pth'):
        checkpoint = torch.load('TorchStateTrainMsft.pth')
        model_conv.load_state_dict(checkpoint)

    #model_conv = torch.nn.DataParallel(model_conv, device_ids=[0,1])
    model_conv = torch.nn.DataParallel(model_conv, device_ids=[0])

    model_conv = model_conv.to(device)

    criterion = nn.CrossEntropyLoss().to(device)

    optimizer_conv = optim.SGD(model_conv.module.fc.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)

    # Decay LR by a factor of 0.94/0.96 every epoch
    exp_lr_scheduler = lr_scheduler.ExponentialLR(optimizer_conv, gamma=0.94)

    model_conv = train_model(model_conv, criterion, optimizer_conv,
                            exp_lr_scheduler, num_epochs=67)

    #torch.save(model_conv.state_dict(), 'D:/Temp/TorchStateFinal.pth')
    # Plot training and validation loss
    graph_train_losses = torch.tensor(train_losses, device='cpu')
    graph_val_losses = torch.tensor(val_losses, device='cpu')

    plt.plot(graph_train_losses, label='Training Loss')
    plt.plot(graph_val_losses, label='Validation Loss')
    plt.title('Loss over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Plot training and validation accuracy
    graph_train_accs = torch.tensor(train_accs, device='cpu')
    graph_val_accs = torch.tensor(val_accs, device='cpu')

    plt.plot(graph_train_accs, label='Training Accuracy')
    plt.plot(graph_val_accs, label='Validation Accuracy')
    plt.title('Accuracy over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
