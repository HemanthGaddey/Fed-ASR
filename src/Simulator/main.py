import multiprocessing, os, psutil, traceback
from multiprocessing import Pool
import time, random
import concurrent.futures
import torch
from torch.utils.data import Dataset
import sys
import signal
import ray 

sys.path.append('..')

from Simulator.Failures import client_fail

from Client.RayLocalClient import RayLocalClient
from Server.RayLocalServer import RayLocalServer
from RayActors.ClientModelsActor import Client_models_actor
from RayActors.ClientStatusActor import Client_status_actor
from Client.Client import FLClient
from Server.server import FLServer

from logging import ERROR, INFO
from typing import Any, Dict, List, Optional, Type, Union
from multiprocessing import Process, Queue, Pool, Manager

class DirichletDataset(Dataset):
    def __init__(self, alpha, cifar_dataset):
        self.alpha = alpha
        self.cifar_dataset = cifar_dataset
        self.dirichlet_dist = torch.distributions.Dirichlet(alpha)

    def __len__(self):
        return len(self.cifar_dataset)

    def __getitem__(self, idx):
        image, label = self.cifar_dataset[idx]
        proportions = self.dirichlet_dist.sample()
        return {'image': image, 'label': label, 'proportions': proportions}


# def set_cpu_affinity(client_id):
#     # Get the number of available CPU cores
#     num_cores = psutil.cpu_count()
#     print(num_cores)
#     # Calculate the CPU core index to which the process should be bound
#     cpu_core_index = client_id % num_cores

#     # Set the CPU affinity of the current process
#     process = psutil.Process(os.getpid())
#     print(process)
#     process.cpu_affinity([cpu_core_index])

# def create_client(data):
# #     set_cpu_affinity(data[0])
#     # print(data)
#     print(f"Creating Client:{data[0]}")
#     try:
#         client = FLClient(id=data[0], port=data[2], name=data[1], dataset_dir=data[3], agg_strategy=data[4], fail_round=data[5])
#     except Exception as e:
#           print(f"\033[31m Failed Creating Client:{data[0]}\033[0m",", reason:", e)
#           traceback_info = traceback.format_exc()
#           print(f"\033[31m TRACEBACK:{traceback_info}\033[0m")
# #     train_model(data)
#     client.train(data=data)
#     return f"Client:{data[0]} work is done!"

# def create_server(name, port, num_clients, min_clients, strategy, cfl):     
#         # server = FLServer(name, port, num_clients, strategy)
#         try:
#             server = FLServer(name, port, num_clients, min_clients, strategy, cfl)
#         except Exception as e:
#           print(f"\033[31m Failed Creating Server! \033[0m",", reason:", e)
#           traceback_info = traceback.format_exc()
#           print(f"\033[31m TRACEBACK:{traceback_info}\033[0m")
#           print()


def start_simulation(
        *,
        name: str,
        num_clients: Optional[int] = None,
        fl_rounds: int,
        port: int,
        dataset_dir: str,
        agg_strategy: str,
        min_clients: int
):
        # for i in range(fl_rounds):
        logdir_path=f"../Log/{name}"
        if not os.path.exists(f"../Log/{name}"):
            os.makedirs(logdir_path)
        if not os.path.exists(f"../Log/{name}/saved_models"): 
            os.makedirs(f"../Log/{name}/saved_models")
        print(f"experiment log files will be in {logdir_path} folder")
        
        for i in range(num_clients):
                metrics_file = f"../Log/{name}/Client-{i}.txt"
                print(metrics_file)
                with open(metrics_file, 'w') as file:
                        print(f"metrics file for Client:{i} is created at {metrics_file}")
                        file.write(f'Model training metrics of Client{i} in experiment named {name} \n')
        cfl,num_failures = client_fail(num_clients=num_clients)
        cfl=[-1]*num_clients
        
        input=[name, port]
        print('Creating Client model ref and Client status ref')
        client_status_actor_ref = Client_status_actor.remote(num_clients,min_clients)
        client_models_actor_ref = Client_models_actor.remote(num_clients,min_clients)

        FLClientActor = ray.remote(FLClient)
        FLServerActor = ray.remote(FLServer)

        print('Creating Server ...')
        # name, port, num_clients, min_clients, strategy, failed_clients_info
        server = FLServerActor.remote(num_clients, min_clients, client_status_actor_ref, client_models_actor_ref, num_rounds=5)
        
        
        print('Creating Client Pool ...')
        print(f'\033[96m Failure Simulation: {cfl} \033[00m')
        clients = [FLClientActor.remote(i,name, dataset_dir, server, client_status_actor_ref,client_models_actor_ref, fail_round=-1, device='cuda') for i in range(num_clients)]
        
        ray.wait([client.loop.remote() for client in clients])

if __name__=="__main__":
    name = 't1'#input("enter experiment name:\n")
    
    agg_strategy = 'fedavg'#input("enter aggregation mode [fedavg/fedbn/fedasr]:")
    dataset_dir = "DIGITS" #"Dirichlet"

    multiprocessing.set_start_method('spawn')
    start_simulation(num_clients=5, fl_rounds=4, name=name, min_clients=5, port=8800, dataset_dir=dataset_dir, agg_strategy=agg_strategy)