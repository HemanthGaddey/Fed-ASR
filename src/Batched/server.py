from multiprocessing import Process, current_process
#from .BaseServer import AbstractServerClass
import time, traceback
import sys
import threading
from datetime import datetime
from queue import Queue
#from RayLocalServer import RayLocalServer
from concurrent import futures
import ray


#aggregation strategy
#from Server.Aggregation.FedAvg import FedAvg
from FedAvg_New import Fedavg
from FedBN import Fedbn_New

import torch
import digit_model #, simple_model, vgg16, custom_model, 

# import io
# sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
# sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

@ray.remote
class FLServer():
    def __init__(self, num_clients, min_clients, client_status_actor_ref, client_models_actor_ref, num_rounds=5, preserve=None, strategy=None, failed_clients_info=None):
        #super().__init__(num_clients, min_clients, num_rounds, client_status_actor_ref, client_models_actor_ref)
        self.client_status_actor_ref = client_status_actor_ref
        self.client_models_actor_ref = client_models_actor_ref
        self.min_client_agg = min_clients
        self.num_clients = num_clients
        self.num_rounds = num_rounds
        self.done = False
        
        self.t = time.time()
        print(time.time()-self.t, 'Started Server Instance!',self.t)
        self.fl_round=0
        self.preserve = preserve
        self.min_clients=min_clients
        self.strategy= strategy
        self.failed_clients_info = failed_clients_info
        # -------------------------select the model here----------------------
        # self.model = custom_model.CNN()
        # self.model = simple_model.SimpleModel()
        self.model = digit_model.DigitModel()
        # self.model = vgg16.VGG16(10)
        # -------------------------------------------------------------------
        self.glob_params = self.model.parameters()

        self.model.to(torch.device('cpu'))
        print(time.time()-self.t, 'Initializing Models')
        for i in range(self.num_clients):
            ray.get(self.client_models_actor_ref.put.remote(i,{'response':'Hi', 'lr':0.1, 'epoch':1, 'momentum': 0.9, 'done':self.done, 'global_epoch':self.fl_round, 'model':self.model}))
        print(time.time()-self.t, 'Initialized Client models, --waiting for aggregation--')

    def get_models(self, idl):
        print(time.time()-self.t, f'idl = {idl}, Hence getting models now')
        config = ray.get(self.client_models_actor_ref.flush.remote(idl))
        models = {i: config[i]['model'] for i in config.keys()}
        print(time.time()-self.t, 'Got models: {models}, models.values(): {models.values()}')
        return config, models
    
    def send_models(self, idl, agg_model):
        if(self.fl_round>=self.num_rounds):
            self.done=True
        for id in idl:
            print(time.time()-self.t, f'Sending model of id:{id}')
            ray.get(self.client_models_actor_ref.put.remote(id,{'response':'Hi', 'lr':0.1, 'epoch':1, 'momentum': 0.9, 'done':self.done, 'global_epoch':self.fl_round, 'model':agg_model}))
            ray.get(self.client_status_actor_ref.put.remote(id,1,None))
        print(time.time()-self.t, f"Finished Sending/Updating models")
        return
    
    def aggregate(self, config, selected_models):
        assert type(selected_models)==type({})
        selected_models = selected_models.values()
        model=None
        print(datetime.now(), "Starting aggregation ")
        try:
            model=Fedavg(selected_models, self.model)#, self.model, self.agg_client_models, selected_model_ids)
            print(f"\033[34m {datetime.now()} self.model is updated via fedavg\033[0m")
            
            # if self.strategy == 'fedavg':
            #     model=Fedavg(selected_models)#, self.model, self.agg_client_models, selected_model_ids)
            #     print(f"\033[34m {datetime.now()} self.model is updated via fedavg\033[0m")
            # elif self.strategy == "fedbn":
            #     num = len(selected_models)
            #     model, self.agg_client_models = Fedbn_New(self.model, selected_models, [1/num]*num)
            # else:
            #     # self.model, self.agg_client_models  = FedASR(selected_models, self.model, self.failed_clients, self.min_clients, self.client_models, [1/num]*num)
            #     pass

            print("===========================================Done with aggregation")
        except Exception as e:
            print(f"\033[31m Failure while aggregating\033[0m")
            error = traceback.print_exc()
            print(f"\033[31m ERROR: {e} \033[0m")
            print(f"\033[31m Traceback: {error} \033[0m")
        self.fl_round+=1
        return model

    def start(self, idl):
        # Get client models
        config, models = self.get_models(idl)

        # Aggregate
        try:
            agg_model = self.aggregate(config, models)
        except Exception as e:
                print(f"\033[31m Failed Server Aggregation\033[0m",", reason:", e)
                traceback_info = traceback.format_exc()
                print(f"\033[31m TRACEBACK:{traceback_info}\033[0m")

        # Send Clients, models
        self.send_models(idl, agg_model) 
