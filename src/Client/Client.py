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

sys.path.append('..')

from Client.RayLocalClient import RayLocalClient
from torch.utils.data import DataLoader, TensorDataset

from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms

from Models.simple_model import SimpleModel
from Models.vgg16 import VGG16
from .BaseClient import AbstractClientClass
from RayActors.ClientModelsActor import Client_models_actor
from RayActors.ClientStatusActor import Client_status_actor

#models
from Models import simple_model, custom_model, digit_model

import signal
import os
import torch
import ray

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

#@ray.remote
class FLClient(RayLocalClient):
    def __init__(self, id, name, dataset_dir, server_actor_ref, client_status_actor_ref, client_models_actor_ref, fail_round, device='cuda'):
        super().__init__(id, server_actor_ref, client_status_actor_ref, client_models_actor_ref)
        self.name = name
        self.device=torch.device(device)
        self.id = id
        self.server_actor_ref = server_actor_ref
        self.client_status_actor_ref = client_status_actor_ref
        self.client_models_actor_ref = client_models_actor_ref
        self.local_round=0
        # -------------------------select the model here----------------------
        # self.model = custom_model.CNN()
        # self.model = simple_model.SimpleModel()
        # self.model = vgg16.VGG16(10)
        # self.model = VGG16(num_classes=10)
        self.model = digit_model.DigitModel()
        # ------------------------Dataset and other params--------------------
        self.best_val=0
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

        # Training loop
        print(f"\033[33m Training started in Client:{self.id}.\033[0m") 
        num_epochs=config['epoch']
        model_log_name = f"Client:{self.id}-{self.local_round}"
        for epoch in range(num_epochs):  

            model.train()
            running_loss=0.0
            correct = 0
            total = 0
            for inputs, labels in self.train_dataloader:
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

        return model

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

        log_training_progress(self.name,f'Client {self.id} [global]',0, 0, 0,val_loss=val_loss, val_acc=val_accuracy,global_epoch=config['global_epoch'])
        global_eval=f"../Log/{self.name}/global_model_eval.txt"
        with open(metrics_file, 'a') as file:
            date_time=datetime.now()
            file.write(f'---------------Client {self.id}: {date_time} round evaluation - val_loss: {val_loss}, val_accuracy: {val_accuracy}%--{correct_val, total_val}---------- \n')
        with open(global_eval, 'a') as file:
            date_time=datetime.now()
            file.write(f'---------------Client {self.id}: {date_time} round evaluation - val_loss: {val_loss}, val_accuracy: {val_accuracy}%------------ \n')

        if(config['done'] != True):
            self.train(config=config, model=model)
        else:   print(f"Client {self.id}: Training is Completed ! 🥳😁")