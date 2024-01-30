#!/usr/bin/env python3

from models.convlstm import ConvLSTMModel
import numpy as np
import torch 
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import datetime as dt

from utils_ import get_unique_elements
from sklearn.model_selection import train_test_split

# Datetime format 
dt_format = '%Y_%m_%d__%H_%M_%S'

# Load the dataset
features =  np.load('data/features.npy')
labels   =  np.load('data/labels.npy')
paths    =  np.load('data/paths.npy')

unique_classes = get_unique_elements(labels)

#Params
batch_size=32
learning_rate = 1e-3
num_epochs = 10

# Load the model
# dev= torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ConvLSTMModel(len(unique_classes)).float()


features = torch.tensor(features, dtype=torch.float32)
labels = torch.tensor(labels, dtype=torch.long)

one_hot_encoded_labels = F.one_hot(labels,num_classes= len(unique_classes))

features_train, features_test, labels_train, labels_test = train_test_split(features,one_hot_encoded_labels,test_size=0.2,random_state=42)

train_dataset = TensorDataset(features_train,labels_train)
train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) 

test_dataset  = TensorDataset(features_test,labels_test)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=True) 

criterion = torch.nn.CrossEntropyLoss()
optimizer =  optim.Adam(model.parameters(),lr=learning_rate)

for epoch  in range(num_epochs):
    model.train()
    train_loss = 0.0

    for inputs, labels  in train_loader:
        optimizer.zero_grad()

        #Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, torch.max(labels,1)[1])

        #Backward pass and optimize
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss/len(train_loader)}")


#Evaluation Loop
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == torch.max(labels, 1)[1]).sum().item()

    print(f'Accuracy on the test set: {100 * correct / total}%')

# Saving the Model  
current = dt.datetime.now()
curent_string = dt.datetime.strftime(current,dt_format)

torch.save(model.state_dict(), f'trained_models/convlstm_model_{curent_string}.pth')



