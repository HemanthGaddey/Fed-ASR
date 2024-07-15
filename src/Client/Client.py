# model_training.py
import os
from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np
import traceback

import time
import torch.optim as optim
from datetime import datetime
import sys
sys.path.append('../')
from torch.utils.data import DataLoader, TensorDataset

from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms

from Models.simple_model import SimpleModel
from Models.vgg16 import VGG16
from .BaseClient import AbstractClientClass
from typing import List
import grpc, multiprocessing
from concurrent import futures
import proto.cbsp_pb2 as cbsp_pb2
import proto.cbsp_pb2_grpc as cbsp_pb2_grpc
from utils.CommunicationService import CommunicationService
from utils.ServerMessage import ServerMessage
from utils.ClientMessage import ClientMessage

#models
from Models import simple_model, custom_model, digit_model

import signal
import os
import torch

from torch.utils.tensorboard import SummaryWriter

# Define a function to log training progress
def log_training_progress(dir_name, model_name, epoch, tr_acc, tr_loss, val_acc, val_loss, global_epoch=-1):
    log_dir = os.path.join(f'../Log/{dir_name}/tensorboard_logs', model_name)
    os.makedirs(log_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=log_dir)

    if(global_epoch>-1):
        writer.add_scalar('Global Model Val_Loss', val_loss, global_epoch)
        writer.add_scalar('Global Model Val_Accuracy', val_acc, global_epoch)
    else:
        writer.add_scalar('Tr_Loss', tr_loss, epoch)
        writer.add_scalar('Tr_Accuracy', tr_acc, epoch)
        writer.add_scalar('Val_Loss', val_loss, epoch)
        writer.add_scalar('Val_Accuracy', val_acc, epoch)

    writer.close()


class FLClient(AbstractClientClass):
    def __init__(self, id, port, name, dataset_dir, agg_strategy, fail_round, device='cuda'):
        super().__init__(id)
        time.sleep(7)
        self.name = name
        self.device=torch.device(device)
        # self.id = id
        self.local_round=0
        # -------------------------select the model here----------------------
        # self.model = custom_model.CNN()
        # self.model = simple_model.SimpleModel()
        # self.model = vgg16.VGG16(10)
        self.model = digit_model.DigitModel()
        # ------------------------------------------------------------------
        self.best_val=0
        # self.model = VGG16(num_classes=10)
        self.dataset_dir=dataset_dir
        self.fail_round = fail_round

        # transform = transforms.Compose([
        # transforms.Resize((224, 224)),  
        # transforms.ToTensor(),           
        # transforms.Normalize(           
        #     mean=[0.485, 0.456, 0.406],   # Mean of ImageNet dataset
        #     std=[0.229, 0.224, 0.225]     # Standard deviation of ImageNet dataset
        # )])
        # self.testset = CIFAR10("./dataset", train=False, download=True, transform=transform)
        # self.test_loader = DataLoader(self.testset, batch_size=16, shuffle=True)

        self.tr_data=torch.load(f"../Data/{self.dataset_dir}/Train/train_data_{self.id}.pth")
        self.val_data=torch.load(f"../Data/{self.dataset_dir}/Validation/val_data_{self.id}.pth")
        
        self.train_dataloader = DataLoader(self.tr_data, batch_size=32, shuffle=True)
        self.val_dataloader = DataLoader(self.val_data, batch_size=32, shuffle=False)

        options = [
        ('grpc.max_receive_message_length', 1024 * 1024 * 1000)  # Adjust the size as needed
        ] 
        channel = grpc.insecure_channel(f'localhost:{port}', options=options)  # Replace with the server address
        self.stub = cbsp_pb2_grpc.CommunicationServiceStub(channel)
        self.cm = ClientMessage(cbsp_pb2)
        d = self.cm.serializeGetParametersMsg({'client id': self.id})
        self.sm = ServerMessage(cbsp_pb2)
        response = self.stub.BidirectionalStream(d)
        info, _, self.model = self.sm.deserializeSendParametersMsg(response, self.model)
        # self.model = self.model.to(self.device)
        print(f"Client {self.id}: <Received from Server> -{info}, client can communicate with the server")
        print(f"\033[105m Client: {id} got initialized, training will happen in {self.device} \033[0m")
        if info['response']=="Hi":
            # print(f"Client {id} starting training")
            self.train(config=info, model=self.model)

    def send_params(self, model):
        print(f"\033[33m {datetime.now()} Sending Client-{self.id} parameters to server \033[0m")

        try:
            request = self.cm.serializeSendParametersMsg({f'id':{self.id}}, model)
            response = self.stub.BidirectionalStream(request)
            d,s = self.sm.deserializeNormalResponseMsg(response)
            print(f"\033[33m {datetime.now()} Client-{self.id} sent parameters-1, {d}, {s}, requesting for aggregated parameters \033[0m")
        except Exception as e:
            print(f"\033[31m {datetime.now()} Failed sending params-1 \033[0m",f"Client ID -{self.id}",", reason:", e)
            traceback_info = traceback.format_exc()
            print(f"\033[31m {datetime.now()} TRACEBACK:{traceback_info}\033[0m",)
        print(f"{datetime.now()} Client-{self.id} completed request-1")
        time.sleep(10)
        try:
            request = self.cm.serializeGetConfigMsg({'id':self.id})
            print("2-1")
            response = self.stub.BidirectionalStream(request)
            print("2-2")
            config, params, model = self.sm.deserializeSendParametersMsg(response, self.model)
            print(config)
            print(f"{datetime.now()} Client-{self.id} received aggregated parameters ")
            self.evaluate(config, model)
        except Exception as e:
            print(f"\033[31m {datetime.now()} Failed sending params-2 \033[0m",f"Client ID -{self.id}",", reason:", e)
            traceback_info = traceback.format_exc()
            print(f"\033[31m {datetime.now()} TRACEBACK:{traceback_info}\033[0m",)

    def signal_handler(signum, frame):
        print(f"Process {multiprocessing.current_process().pid} received signal {signum}. Exiting...")
        raise SystemExit
    
    def train(self,config: dict, model):
        print(f'{datetime.now()} CLIENT-{self.id} is ACTIVE and starting TRAINING!')
        model = model.to(self.device)
        # for crashing the client during the simulation
        if(self.local_round==self.fail_round):
            print(f'\033[91m Client {self.id} Exiting in round {self.local_round} due to encountering a failure! \033[00m')
            with open(metrics_file, 'a') as file:
                date_time=datetime.now()
                file.write(f'[{date_time}] - Client {self.id} Exiting in round {self.local_round} due to encountering a failure! ')
            sys.exit(0)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        metrics=nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=config['lr'], momentum=config['momentum'])
        # print("here!")
        # print(f"../Data/{self.dataset_dir}/Train/train_data_{self.id}.pth")
        # tr_data=torch.load(f"../Data/{self.dataset_dir}/Train/train_data_{self.id}.pth")
        # # print(tr_data)
        # val_data=torch.load(f"../Data/{self.dataset_dir}/Validation/val_data_{self.id}.pth")
        # # dataset = TensorDataset(X, y)
        # train_dataloader = DataLoader(tr_data, batch_size=16, shuffle=True)
        # val_dataloader = DataLoader(val_data, batch_size=16, shuffle=True)
        metrics_file = f"../Log/{self.name}/Client-{self.id}.txt"
        with open(metrics_file, 'a') as file:
                date_time=datetime.now()
                # file.write(f'FL round - {data[2]}')
        print(f"\033[33m Training started in Client:{self.id}.\033[0m") 
        # Training loop
        # print(config['epoch'], model)
        num_epochs=config['epoch']
        # print(self.id, num_epochs)
        model_log_name = f"Client:{self.id}-{self.local_round}"
        for epoch in range(num_epochs):  # 5 epochs for illustration
            model.train()
            running_loss=0.0
            correct = 0
            total = 0
            for inputs, labels in self.train_dataloader:
                # print(labels) 
                # try:
                #     inputs, labels= inputs.to(device), labels.to(device)
                # except Exception as e:
                #     print('error while mving labels to gpu', e)
                # # print(labels)
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = metrics(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1e-2)
                optimizer.step()

                running_loss+=loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            model.eval()
            correct_val=0
            total_val=0
            val_loss=0
            with torch.no_grad():
                for inputs, labels in self.val_dataloader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total_val += labels.size(0)
                    correct_val += (predicted == labels).sum().item()
                    loss = metrics(outputs, labels)
                    val_loss+=loss

            # Calculate validation accuracy
            val_accuracy = 100 * correct_val / total_val
            val_loss = val_loss / len(self.val_dataloader)
            accuracy = 100 * correct / total
            average_loss = running_loss / len(self.train_dataloader)
            if val_accuracy>self.best_val:
                self.best_val = val_accuracy
                model.state_dict()
                torch.save(model.state_dict(),f"../Log/{self.name}/saved_models/client-{self.id}_bestmodel.pth")
            e=epoch+1
            log_training_progress(self.name, model_log_name, e, accuracy, average_loss, val_accuracy, val_loss)
            
            with open(metrics_file, 'a') as file:
                date_time=datetime.now()    
                file.write(f'{date_time} Epoch {epoch + 1}/{num_epochs}, Train_Avg_Loss: {average_loss:.3f}, Train Accuracy: {accuracy}%, Validation Accuracy: {val_accuracy}%, Validation_Avg_Loss: {val_loss:.3f} \n')
            
            torch.cuda.empty_cache()
        self.local_round+=1
        self.send_params(model)

      
    def evaluate(self, config, model):
    
        metrics=nn.CrossEntropyLoss()

        metrics_file = f"../Log/{self.name}/Client-{self.id}.txt"
        num_epochs=config['epoch']
        print(f"\033[33m Evaluation started in Client:{self.id}.\033[0m", "val_epochs:", num_epochs)
        
        # Training loop
        model.eval()
        correct_val=0
        val_loss=0
        total_val=0
        with torch.no_grad():
            for inputs, labels in (self.val_dataloader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = model(inputs)
                loss=metrics(outputs, labels)
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                val_loss+=loss
                correct_val += (predicted == labels).sum().item()
            torch.cuda.empty_cache()   
        val_loss=val_loss/len(self.val_dataloader)
        val_accuracy = 100 * correct_val / total_val

        # correct_test=0
        # test_loss=0
        # total_test=0
        # with torch.no_grad():
        #     for inputs, labels in (self.test_loader):
        #         inputs, labels = inputs.to(self.device), labels.to(self.device)
        #         outputs = model(inputs)
        #         loss=metrics(outputs, labels)
        #         _, predicted = torch.max(outputs.data, 1)
        #         total_test += labels.size(0)
        #         test_loss+=loss
        #         correct_test += (predicted == labels).sum().item()
                
        # test_loss=test_loss/len(val_dataloader)
        # test_accuracy = 100 * correct_test / total_test

        log_training_progress(self.name,f'Client {self.id} [global]',0, 0, 0,val_loss=val_loss, val_acc=val_accuracy,global_epoch=config['global_epoch'])

        global_eval=f"../Log/{self.name}/global_model_eval.txt"

        with open(metrics_file, 'a') as file:
            date_time=datetime.now()
            file.write(f'---------------Client {self.id}: {date_time} round evaluation - val_loss: {val_loss}, val_accuracy: {val_accuracy}%--{correct_val, total_val}---------- \n')

        with open(global_eval, 'a') as file:
            date_time=datetime.now()
            file.write(f'---------------Client {self.id}: {date_time} round evaluation - val_loss: {val_loss}, val_accuracy: {val_accuracy}%------------ \n')
        
        # request = self.cm.serializeGetConfigMsg({'id':self.id,'val accuracy':val_accuracy})
        # response = self.stub.BidirectionalStream(request)
        # config, params, model = self.sm.deserializeSendParametersMsg(response, model)
        
        if(config['done'] != True):
            self.train(config=config, model=model)
        else:   print(f"Client {self.id}: Round is Completed ! 🥳😁")
