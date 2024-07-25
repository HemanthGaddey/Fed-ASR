import time
import numpy as np
import ray
import digit_model #, simple_model, vgg16, custom_model, 
import torch


@ray.remote
class Client_models_actor:
    def __init__(self,num_clients, min_clients):
        init_model = digit_model.DigitModel().to(torch.device('cpu'))
        self.models = {k:{'response':'Hi', 'lr':0.1, 'epoch':1, 'momentum': 0.9, 'done':False, 'global_epoch':0, 'model':init_model} for k in list(range(num_clients))}
        self.num_clients = num_clients
        self.min_clients = min_clients
#         self.server_actor_ref = server_actor_ref
        
    def get_length(self):
        return len(self.models)
    
    def flush(self, idl=[]):
        print(f'flush request, idl={idl}!, sending now')
        #print(f'flush curr models: {self.models}')
    
        if(len(idl)>0):
            models = {k: self.models[k] for k in idl}
            for key in idl:
                self.models.pop(key, None)
        else:
            print('Empty list, flushing all models')
            models = self.models
            self.models = {}
            
        return models
    
    def get(self, index):
        return self.models[index]
    
    def put(self, index, model_dict):
        self.models[index] = model_dict
