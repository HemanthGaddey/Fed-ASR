# model_training.py
import os
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
import sys
sys.path.append('../')
from torch.utils.data import DataLoader, TensorDataset
from Models.simple_model import SimpleModel
from Models.resnet18 import Resnet18
from .BaseClient import AbstractClientClass
from typing import List
import grpc
from concurrent import futures
import proto.cbsp_pb2 as cbsp_pb2
import proto.cbsp_pb2_grpc as cbsp_pb2_grpc
from utils.CommunicationService import CommunicationService
from utils.ServerMessage import ServerMessage
from utils.ClientMessage import ClientMessage
from Models import simple_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FLClient_new(AbstractClientClass):
    def __init__(self, id, port, name):
        super().__init__(id)
        self.name = name
        self.id = id
        self.model = Resnet18()
        options = [
        ('grpc.max_receive_message_length', 1024 * 1024 * 1000)  # Adjust the size as needed
        ]
        # print(f"Client: {id} initializing")
        channel = grpc.insecure_channel(f'localhost:{port}', options=options)  # Replace with the server address
        self.stub = cbsp_pb2_grpc.CommunicationServiceStub(channel)
        self.cm = ClientMessage(cbsp_pb2)
        d = self.cm.serializeGetParametersMsg({'client': self.id})
        self.sm = ServerMessage(cbsp_pb2)
        # Send the client message and get the server response
        # print(f"Client {id} Asking Server for parameters, model:", self.model)
        response = self.stub.BidirectionalStream(d)
        # print(f"Client {id} received response from server")
        info, _, model = self.sm.deserializeSendParametersMsg(response, self.model)
        print(f"Client {self.id}: <Received from Server> -{info}, client can communicate with the server")
        if info['response']=="Hi":
            print(f"Client {id} starting training")
            self.train(config=info, model=self.model)

    def send_params(self, model):
        print(f"Sending Client-{self.id} parameters to server")

        request = self.cm.serializeSendParametersMsg({f'id':{self.id}}, model)
        response = self.stub.BidirectionalStream(request)
        d,s = self.sm.deserializeNormalResponseMsg(response)
        
        print(f"Client {self.id} sent params to server | <Server Response: True )")
        print(f"*****Client {self.id}: Asking for updated params from server*****")

        request = self.cm.serializeGetConfigMsg({'id':self.id})
        response = self.stub.BidirectionalStream(request)
        config, params, model = self.sm.deserializeSendParametersMsg(response, self.model)
        print(f"CLient {self.id} will Sleep now lol...")
       # self.evaluate(config, model)
    
    def train(self,config: dict, model):        
        metrics=nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=config['lr'], momentum=config['momentum'])
        tr_data=torch.load(f"..\\Data\\Train\\train_data_{self.id}.pth")
        val_data=torch.load(f"..\\Data\\Validation\\val_data_{self.id}.pth")
        # dataset = TensorDataset(X, y)
        train_dataloader = DataLoader(tr_data, batch_size=16, shuffle=True)
        val_dataloader = DataLoader(val_data, batch_size=16, shuffle=True)
        metrics_file = f"..\\Log\\{self.name}\\Client-{self.id}.txt"
        with open(metrics_file, 'a') as file:
                date_time=datetime.now()
                # file.write(f'FL round - {data[2]}')
        print(f"Training started in Client:{self.id}.")
        num_epochs=0
        
        self.send_params(model)
        # print(f"Client {self.id} received params from server ")
        # Save the trained model (you may use torch.save() here)
        # print(f"Client:{id} training completed.")
        # return f"Client:{id} training completed."
    
    def evaluate(self, config, model):
        
        val_data=torch.load(f"..\\Data\\Validation\\val_data_{self.id}.pth")
        val_dataloader = DataLoader(val_data, batch_size=16, shuffle=True)
        metrics_file = f"..\\Log\\{self.name}\\Client-{self.id}.txt"
        with open(metrics_file, 'a') as file:
                date_time=datetime.now()
                # file.write(f'FL round - {data[2]}')
        num_epochs=1
        # Training loop
        model.eval()
        correct_val=0
        total_val=0
        with torch.no_grad():
            for inputs, labels in val_dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        val_accuracy = 100 * correct_val / total_val

        with open(metrics_file, 'a') as file:
            date_time=datetime.now()
            file.write(f'Client {self.id}: {date_time} Validation Accuracy: {val_accuracy}% \n')

        
        request = self.cm.serializeGetConfigMsg({'id':self.id,'val accuracy':val_accuracy})
        response = self.stub.BidirectionalStream(request)
        config, params, model = self.sm.deserializeSendParametersMsg(response, model)
        
        if(config['done'] != True):
            self.train(config=config, model=model)
        else:
            print(f"Client {self.id}: Ayipoyindi brooooo! 🥳😁")

def train_model(data):
    # Example: Training logic with dummy data
    model = Resnet18()
    metrics=nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    tr_data=torch.load(f"..\\Data\\Train\\train_data_{data[0]}.pth")
    val_data=torch.load(f"..\\Data\\Validation\\val_data_{data[0]}.pth")
    # dataset = TensorDataset(X, y)
    train_dataloader = DataLoader(tr_data, batch_size=16, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=16, shuffle=True)
    metrics_file = f"..\\Log\\{data[1]}\\Client-{data[0]}.txt"
    with open(metrics_file, 'a') as file:
            date_time=datetime.now()
            file.write(f'FL round - {data[2]}')
    print(f"Training started in Client:{data[0]}.")
    num_epochs=5
    # Training loop
    for epoch in range(num_epochs):  # 5 epochs for illustration
        model.train()
        running_loss=0.0
        correct = 0
        total = 0
        for inputs, labels in train_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = metrics(outputs, labels)
            loss.backward()
            optimizer.step()

        running_loss+=loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        model.eval()
        correct_val=0
        total_val=0
        with torch.no_grad():
            for inputs, labels in val_dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        # Calculate validation accuracy
        val_accuracy = 100 * correct_val / total_val
        accuracy = 100 * correct / total
        average_loss = running_loss / len(train_dataloader)

        with open(metrics_file, 'a') as file:
            date_time=datetime.now()
            file.write(f'{date_time} Epoch {epoch + 1}/{num_epochs}, Average Loss: {average_loss:.3f}, Train Accuracy: {accuracy}%, Validation Accuracy: {val_accuracy}% \n')


    # Save the trained model (you may use torch.save() here)
    print(f"Client:{data[0]} training completed.")

if __name__ == "__main__":
    train_model((0,'one'))
