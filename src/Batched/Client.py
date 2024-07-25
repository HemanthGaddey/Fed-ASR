# model_training.py
import os
from tqdm import tqdm
import torch
import torch.nn as nn

import torch.optim as optim
from datetime import datetime
import sys
import time


from tqdm import tqdm
#from RayLocalClient import RayLocalClient
from torch.utils.data import DataLoader, TensorDataset

import multiprocessing, os, psutil, traceback
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms

# from Models.simple_model import SimpleModel
# from Models.vgg16 import VGG16
#from BaseClient import AbstractClientClass
from ClientModelsActor import Client_models_actor
from ClientStatusActor import Client_status_actor

#models
import digit_model#, simple_model, custom_model, 

import signal
import os
import torch
import ray

from torch.utils.tensorboard import SummaryWriter

# Define a function to log training progress
def log_training_progress(dir_name, model_name, epoch, tr_acc, tr_loss, val_acc, val_loss, global_epoch=-1):
    log_dir = os.path.join(f'..',f'Log',f'{dir_name}',f'tensorboard_logs', model_name)
    try:
        os.makedirs(log_dir, exist_ok=True)
    except Exception as e:
        print(f"\033[31m Directory making error\033[0m",", reason:", e)
        traceback_info = traceback.format_exc()
        print(f"\033[31m TRACEBACK:{traceback_info}\033[0m")

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

@ray.remote(num_gpus=0.2)
class FLClient():
    def __init__(self, id, name, dataset_dir, server_actor_ref, client_status_actor_ref, client_models_actor_ref, fail_round, device):
        #super().__init__(id, server_actor_ref, client_status_actor_ref, client_models_actor_ref)
        
        self.t = time.time()
        print(time.time()-self.t, '\tStarting Client Actions!',self.t)
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
        
        self.train_dataloader = DataLoader(self.tr_data, batch_size=3, shuffle=True)
        self.val_dataloader = DataLoader(self.val_data, batch_size=3, shuffle=False)

        print(f'Client {self.id} initialized with model: {self.model}')

    def get_model(self):
        status = ray.get(self.client_status_actor_ref.get.remote(self.id))
        while(status!=1):
            time.sleep(1)
            print(time.time()-self.t, f'status={status} so waiting')
            status = ray.get(self.client_status_actor_ref.get.remote(self.id))
        print(time.time()-self.t, f'status={status}!! so retreiving model')
        config = ray.get(self.client_models_actor_ref.get.remote(self.id))
        model = config['model']
        ray.get(self.client_status_actor_ref.put.remote(self.id,0,self.server_actor_ref))
        print(time.time()-self.t, f'\tClient {self.id} recieved model!!!')
        return config, model
        
    def push_model(self,model):
        ray.get(self.client_models_actor_ref.put.remote(self.id, {'model':model}))
        ray.get(self.client_status_actor_ref.put.remote(self.id, 2, self.server_actor_ref))
        print(time.time()-self.t, f'\tClient {self.id} pushed the model!!!')

    
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
        model_log_name = f"Client_{self.id}-{self.local_round}"
        for epoch in range(num_epochs):  
            print(f'--------->  In training Epoch {epoch}')
            model.train()
            running_loss=0.0
            correct = 0
            total = 0
            for inputs, labels in self.train_dataloader:
                if(len(labels)==1):
                    print('Emergency!!! only single item to train in batch! - Batch Norm will give error!!')
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
                    if(len(labels)==1):
                        print('Emergency!!! only single item to train in batch! - Batch Norm will give error!!')
                        continue
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
            #torch.cuda.empty_cache()
        
        self.local_round+=1
        print(f'-------> Done Training, local round: {self.local_round}')
        return model.to(torch.device('cpu'))

    def evaluate(self, config, model):
        metrics=nn.CrossEntropyLoss()

        metrics_file = f"../Log/{self.name}/Client-{self.id}.txt"
        num_epochs=config['epoch']
        print(f"\033[33m Evaluation started in Client:{self.id}.\033[0m", "val_epochs:", num_epochs)
        
        model.to(torch.device(self.device))
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

        model.to(torch.device('cpu'))
        # if(config['done'] != True):
        #     self.train(config=config, model=model)
        # else:   print(f"Client {self.id}: Training is Completed ! 🥳😁")

    def loop(self):
        while True:
            # Get model
            config, model = self.get_model()
            
            # Evaluate
            print(time.time()-self.t, f'Started Evaluation, ')#config = {config}')
            self.evaluate(config, model)
            print(time.time()-self.t, 'Finished Evaluation')
            
            # Train+
            try:
                if(config['done'] == True):
                    print(f"Client {self.id}: Training is Completed ! 🥳😁")
                    break
                print(time.time()-self.t, 'Started Training')
                new_model = self.train(config, model)
                self.model = new_model
                print(time.time()-self.t, 'Finished Training')
            except Exception as e:
                print(f"\033[31m Failed Training Client\033[0m",", reason:", e)
                traceback_info = traceback.format_exc()
                print(f"\033[31m TRACEBACK:{traceback_info}\033[0m")
            
            # Push model
            self.push_model(self.model)
