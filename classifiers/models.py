import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse
from tqdm import tqdm

class MLP(nn.Module):
    """ adapted from https://github.com/CUAI/CorrectAndSmooth/blob/master/gen_models.py """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout=.5):
        super(MLP, self).__init__()
        self.lins = nn.ModuleList()
        self.bns = nn.ModuleList()
        if num_layers == 1:
            # just linear layer i.e. logistic regression
            self.lins.append(nn.Linear(in_channels, out_channels))
        else:
            self.lins.append(nn.Linear(in_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
            for _ in range(num_layers - 2):
                self.lins.append(nn.Linear(hidden_channels, hidden_channels))
                self.bns.append(nn.BatchNorm1d(hidden_channels))
            self.lins.append(nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, data):
        x = data
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = F.relu(x, inplace=True)
            x = self.bns[i](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x

class CNN(nn.Module):
    """ adapted from https://medium.com/@nutanbhogendrasharma/pytorch-convolutional-neural-network-with-mnist-dataset-4e8a4265e118 """
    def __init__(self, in_channels, out_channels, hidden_channels, kernel_size,
                  dropout):
      super(CNN, self).__init__()
      self.conv1 = nn.Conv2d(in_channels = 32, 
                             out_channels = hidden_channels, 
                             kernel_size=kernel_size, 
                             stride=1)
      self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1)
      self.dropout1 = nn.Dropout2d(0.25)
      self.dropout2 = nn.Dropout2d(0.5)
      self.fc1 = nn.Linear(144, 128)
      self.fc2 = nn.Linear(128, out_channels)
    

    # x represents our data
    def forward(self, x):
      # Pass data through conv1
      x = self.conv1(x)
      # Use the rectified-linear activation function over x
      x = F.relu(x)

      x = self.conv2(x)
      x = F.relu(x)

      # Run max pooling over x
      x = F.max_pool2d(x, 2)
      # Pass data through dropout1
      x = self.dropout1(x)
      # Flatten x with start_dim=1
      x = torch.flatten(x, 1)
      # Pass data through fc1
      x = self.fc1(x)
      x = F.relu(x)
      x = self.dropout2(x)
      x = self.fc2(x)

      # Apply softmax to x 
      output = F.log_softmax(x, dim=1)
      return output

# class CNN(nn.Module):
#     def __init__(self, in_channels=1, hidden_channels=32, out_channels=1, kernel_size = 3,
#                   dropout = 0.5):
#         super(CNN, self).__init__()
    
#         # Initialize the ConvNet layers
# #         raise NotImplementedError("Your code goes here!")
#         self.conv1 = nn.Conv2d(in_channels=1, out_channels = hidden_channels, 
#                                kernel_size=3, stride=1, padding=1, bias=False)
        
#         self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels = hidden_channels, 
#                                kernel_size=3, stride=1, padding=1, bias=False)
         
#         self.fc = nn.Linear( 7 * 7 * hidden_channels,  out_channels)
        
#     def forward(self, x):
#         x = self.conv1(x)
#         x = F.relu(x, inplace=True)
#         x = F.max_pool2d(x,kernel_size=2, stride=2)
        
#         x = self.conv2(x)
#         x = F.relu(x)
#         x = F.max_pool2d(x,kernel_size=2, stride=2)
#         x = x.view(x.size(0), -1) # flatten all dimensions
# #         print(x.shape)
#         x = self.fc(x)

#         return x
    
    
    
    
#     def __init__(self, in_channels = 1, out_channels=16, kernel_size=5,
#                  dropout=.5):
#         super(CNN, self).__init__()
#         self.conv1 = nn.Sequential(         
#             nn.Conv2d(
#                 in_channels=1,              
#                 out_channels=32,            
#                 kernel_size=3,              
#                 stride=1,                   
#                 padding=2,                  
#             ),                              
#             nn.ReLU(),                      
#             nn.MaxPool2d(kernel_size=2),    
#         )
#         self.conv2 = nn.Sequential(         
#             nn.Conv2d(32, 64, 3, 1, 2),     
#             nn.ReLU(),                      
#             nn.MaxPool2d(2),                
#         )
#         # fully connected layer, output 10 classes
#         self.out = nn.Linear(32 * 7 * 7, 10)
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.conv2(x)
#         # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
#         x = x.view(x.size(0), -1)       
#         output = self.out(x)
#         return output
