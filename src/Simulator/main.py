import multiprocessing, os, psutil, traceback
from multiprocessing import Pool
import time, random
import concurrent.futures
import torch
from torch.utils.data import Dataset
import sys
import signal
sys.path.append('../')

from Simulator.Failures import client_fail

from Server.server import FLServer
from Client.Client import FLClient
from Client.client_new import FLClient_new
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


def set_cpu_affinity(client_id):
    # Get the number of available CPU cores
    num_cores = psutil.cpu_count()
    print(num_cores)
    # Calculate the CPU core index to which the process should be bound
    cpu_core_index = client_id % num_cores

    # Set the CPU affinity of the current process
    process = psutil.Process(os.getpid())
    print(process)
    process.cpu_affinity([cpu_core_index])

def create_client(data):
#     set_cpu_affinity(data[0])
    # print(data)
    print(f"Creating Client:{data[0]}")
    try:
        client = FLClient(id=data[0], port=data[2], name=data[1], dataset_dir=data[3], agg_strategy=data[4], fail_round=data[5])
    except Exception as e:
          print(f"\033[31m Failed Creating Client:{data[0]}\033[0m",", reason:", e)
          traceback_info = traceback.format_exc()
          print(f"\033[31m TRACEBACK:{traceback_info}\033[0m")
#     train_model(data)
    client.train(data=data)
    return f"Client:{data[0]} work is done!"

def create_server(name, port, num_clients, min_clients, strategy, cfl):     
        # server = FLServer(name, port, num_clients, strategy)
        try:
            server = FLServer(name, port, num_clients, min_clients, strategy, cfl)
        except Exception as e:
          print(f"\033[31m Failed Creating Server! \033[0m",", reason:", e)
          traceback_info = traceback.format_exc()
          print(f"\033[31m TRACEBACK:{traceback_info}\033[0m")
          print()

def signal_handler(sig, frame, server_process, pool):
    print('Caught interrupt signal, terminating processes...')
    pool.terminate()
    pool.join()
    server_process.terminate()
    server_process.join()
    sys.exit(0)

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
        print('Creating Server ...')
        # name, port, num_clients, min_clients, strategy, failed_clients_info
        server_process = multiprocessing.Process(target=create_server, args=(name, port, num_clients, min_clients, agg_strategy, cfl))
        server_process.start()

        
        print('Creating Client Pool ...')
        print(f'\033[96m Failure Simulation: {cfl} \033[00m')
        pool=multiprocessing.Pool(processes=num_clients)
        inputs=[[i, name, port, dataset_dir, agg_strategy, cfl[i]] for i in range(num_clients)]
        
        signal.signal(signal.SIGINT, lambda sig, frame: signal_handler(sig, frame, server_process, pool))
        
        # for epoch in range(fl_rounds):     
        pool.map_async(create_client, inputs)
        pool.close()
        pool.join()

        server_process.terminate()
        server_process.join()

if __name__=="__main__":
    name = input("enter experiment name:\n")
    
    agg_strategy = input("enter aggregation mode [fedavg/fedbn/fedasr]:")
    dataset_dir = "DIGITS" #"Dirichlet"

    multiprocessing.set_start_method('spawn')
    start_simulation(num_clients=5, fl_rounds=4, name=name, min_clients=5, port=8800, dataset_dir=dataset_dir, agg_strategy=agg_strategy)