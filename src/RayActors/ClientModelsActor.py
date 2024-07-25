import time
import numpy as np
import ray

@ray.remote
class Client_models_actor:
    def __init__(self,num_clients, min_clients):
        self.models = {k:{'model':0} for k in list(range(num_clients))}
        self.num_clients = num_clients
        self.min_clients = min_clients
#         self.server_actor_ref = server_actor_ref
        
    def get_length(self):
        return len(self.models)
    
    def flush(self, idl=[]):
        print(f'flush request, idl={idl}!, sending now')
        print(f'flush curr models: {self.models}')
    
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