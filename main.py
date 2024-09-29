import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
import torchaudio
import torchaudio.functional as AF
import matplotlib.pyplot as plt
from torchmetrics import Accuracy

# Define directories for your dataset
# train_data_dir = '../Dataset/Train'
# test_data_dir = '../Dataset/Test'

# Create a dictionary mapping from class label to index
# def create_label_map(directory):
#     labels = sorted([d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))])
#     return {label: idx for idx, label in enumerate(labels)}

# label_map = create_label_map(train_data_dir)
labels = ['sheila', 'right', 'house', 'five', 'left',
              'dog', 'visual', 'seven', 'up', 'forward',
                'marvin', 'go', 'one', 'learn', 'on', 'six',
                  'off', 'down', 'eight', 'stop', 'zero', 'four',
                    'nine', 'three', 'backward', 'wow', 'tree', 'bed',
                      'follow', 'cat', 'happy', 'no', 'two', 'bird', 'yes']
labels = sorted(labels)
label_map = {label: idx for idx, label in enumerate(labels)}
# print(label_map)
num_class = len(label_map)

# Custom Dataset class for loading audio files
class AudioDataset(Dataset):
    def __init__(self, data_dir, label_map, transform=None):
        self.data_dir = data_dir
        self.label_map = label_map
        self.transform = transform
        self.filepaths = []
        self.labels = []
        
        # Read filepaths and labels
        for label in os.listdir(data_dir):
            label_dir = os.path.join(data_dir, label)
            if os.path.isdir(label_dir):
                for file_name in os.listdir(label_dir):
                    if file_name.endswith('.wav'):  # Assumes .wav files
                        self.filepaths.append(os.path.join(label_dir, file_name))
                        self.labels.append(self.label_map[label])

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        file_path = self.filepaths[idx]
        label = self.labels[idx]
        
        # Load audio file
        waveform, sample_rate = torchaudio.load(file_path)
        
        # Optionally downsample the audio (as done before)
        waveform = AF.resample(waveform, orig_freq=sample_rate, new_freq=8000)
        
        return waveform, label

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# Define a collate function to pad sequences and create batches
def collate_fn(batch):
    data = [item[0][0] for item in batch]  # Extract the waveform from each item in batch
    data = pad_sequence(data, batch_first=True)  # Pad sequences
    data = data.unsqueeze(1)  # Add channel dimension
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)  # Extract labels
    return data, labels

# Load the dataset
# train_dataset = AudioDataset(train_data_dir, label_map)
# test_dataset = AudioDataset(test_data_dir, label_map)

# Create DataLoader
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
# test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

# Define the model
class M5(nn.Module):
    def __init__(self, n_input=1, n_output=num_class, stride=16, n_channel=32):
        super().__init__()
        self.conv1 = nn.Conv1d(n_input, n_channel, kernel_size=80, stride=stride)
        self.bn1 = nn.BatchNorm1d(n_channel)
        self.pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(n_channel, n_channel, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(n_channel)
        self.pool2 = nn.MaxPool1d(4)
        self.conv3 = nn.Conv1d(n_channel, 2 * n_channel, kernel_size=3)
        self.bn3 = nn.BatchNorm1d(2 * n_channel)
        self.pool3 = nn.MaxPool1d(4)
        self.conv4 = nn.Conv1d(2 * n_channel, 2 * n_channel, kernel_size=3)
        self.bn4 = nn.BatchNorm1d(2 * n_channel)
        self.pool4 = nn.MaxPool1d(4)
        self.fc1 = nn.Linear(2 * n_channel, n_output)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool2(x)
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.pool3(x)
        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        x = self.pool4(x)
        x = F.avg_pool1d(x, x.shape[-1])
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        return x.squeeze(1)

# Initialize the model, optimizer, and loss function
model = M5()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
loss_fn = nn.CrossEntropyLoss()

# Define training and evaluation functions
# def train_one_epoch(model, train_loader, loss_fn, optimizer):
#     model.train()
#     loss_train = AverageMeter()
#     acc_train = Accuracy(task="multiclass", num_classes=num_class)
#     for inputs, targets in train_loader:
#         outputs = model(inputs)
#         loss = loss_fn(outputs, targets)
#         loss.backward()
#         nn.utils.clip_grad_norm_(model.parameters(), 1)
#         optimizer.step()
#         optimizer.zero_grad()
#         loss_train.update(loss.item())
#         acc_train(torch.argmax(outputs, dim=1), targets.int())
#     return model, loss_train.avg, acc_train.compute().item()

# def evaluate(model, loader, loss_fn):
#     model.eval()
#     loss_valid = AverageMeter()
#     acc_valid = Accuracy(task="multiclass", num_classes=num_class)
#     with torch.no_grad():
#         for inputs, targets in loader:
#             outputs = model(inputs)
#             loss = loss_fn(outputs, targets)
#             loss_valid.update(loss.item())
#             acc_valid(torch.argmax(outputs, dim=1), targets.int())
#     return loss_valid.avg, acc_valid.compute().item()

# Training loop
# num_epochs = 15
# loss_train_hist, acc_train_hist, loss_test_hist, acc_test_hist = [], [], [], []

# for epoch in range(num_epochs):
#     model, loss_train, acc_train = train_one_epoch(model, train_loader, loss_fn, optimizer)
#     loss_test, acc_test = evaluate(model, test_loader, loss_fn)
    
#     loss_train_hist.append(loss_train)
#     acc_train_hist.append(acc_train)
#     loss_test_hist.append(loss_test)
#     acc_test_hist.append(acc_test)
    
#     print(f'Epoch {epoch+1}/{num_epochs}')
#     print(f'Train Loss: {loss_train:.4f}, Train Acc: {acc_train:.4f}')
#     print(f'Test Loss: {loss_test:.4f}, Test Acc: {acc_test:.4f}')

# Save the trained model
# torch.save(model.state_dict(), 'speech_classification_model.pth')

# Load the trained model
model = M5()  # Initialize model
model.load_state_dict(torch.load('speech_classification_model.pth'))  # Load saved weights
model.eval()  # Set the model to evaluation mode

# Define a function for prediction on a single audio file
def predict_audio_file(file_path, model, label_map):
    # Load and process the audio file
    waveform, sample_rate = torchaudio.load(file_path)
    waveform = AF.resample(waveform, orig_freq=sample_rate, new_freq=8000)
    waveform = waveform.unsqueeze(0)  # Add batch dimension
    
    # Get predictions
    with torch.no_grad():
        output = model(waveform)
        predicted_class_index = torch.argmax(output, dim=1).item()
        
    # Convert index to label
    label_map_reverse = {v: k for k, v in label_map.items()}
    predicted_label = label_map_reverse[predicted_class_index]
    
    return predicted_label

# Example usage: Predict the label of an audio file
score = 0
title = 0
for test in os.listdir('./test/'):
    title += 1
    new_file_path = f'./test/{test}'
    # print(new_file_path)
    predicted_label = predict_audio_file(new_file_path, model, label_map)
    text = test.split('_')
    # print(text[0])
    # print(f'The predicted label for {new_file_path} is: {predicted_label}')
    print(f'Answer is {text[0]} : Predicted is {predicted_label}',end='')
    if(text[0].lower() == str(predicted_label).lower()):
        print(' 1')
        score = score + 1
    else:
        print('')
print(f'{score}/{title} = {int((score/title)*100)}%')