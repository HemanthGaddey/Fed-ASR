import time
import numpy as np
import ray
import multiprocessing, os, psutil, traceback
import torch

#@ray.remote
class RayLocalClient:
    def __init__(self, id, server_actor_ref, client_status_actor_ref, client_models_actor_ref):
        self.id = id
        self.server_actor_ref = server_actor_ref
        self.client_status_actor_ref = client_status_actor_ref
        self.client_models_actor_ref = client_models_actor_ref
        self.model = None
        print(f'Client {self.id} initialized with model: {self.model}')
        
        self.t = time.time()
        print(time.time()-self.t, '\tStarting Client Actions!',self.t)

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
        
    def evaluate(self):
        pass
    
    def train(self):
        pass
    
    def loop(self):
        while True:
            # Get model
            config, model = self.get_model()
            
            # Evaluate
            print(time.time()-self.t, f'Started Evaluation, ')#config = {config}')
            #self.evaluate(config, model)
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
    
    def test(self):
        pass