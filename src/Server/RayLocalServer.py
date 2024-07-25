import time
import numpy as np
import ray

#@ray.remote
class RayLocalServer:
    def __init__(self, num_clients, min_client_agg, num_rounds, client_status_actor_ref, client_models_actor_ref):
        self.client_status_actor_ref = client_status_actor_ref
        self.client_models_actor_ref = client_models_actor_ref
        self.min_client_agg = min_client_agg
        self.num_clients = num_clients
        self.num_rounds = num_rounds
        self.done = False
        
        self.t = time.time()
        print(time.time()-self.t, 'Started Server Instance!',self.t)
        print(time.time()-self.t, 'Initializing Models')
        for i in range(self.num_clients):
            ray.get(self.client_models_actor_ref.put.remote(i,0))
        print(time.time()-self.t, 'Initialized Client models, --waiting for aggregation--')
    

    def get_models(self, idl):
        print(time.time()-self.t, f'idl = {idl}, Hence getting models now')
        config = ray.get(self.client_models_actor_ref.flush.remote(idl))
        models = {i: config[i]['model'] for i in config.keys()}
        print(time.time()-self.t, f'Got models: {models}, models.values(): {models.values()}')
        return config, models
    
    def aggregate(self, models):
        print(time.time()-self.t, '\t', f'models: {models}; beginning aggregation')
        if(len(models)>0):
            agg_model = np.mean(list(models.values()))
        else:
            print('In aggregate function, len(models) == 0, Hence keeping agg_model as 0')
            agg_model = 0
        print(time.time()-self.t, '\t', f'Aggregated models {models}, {agg_model}')
        return agg_model
    
    def send_models(self, idl, agg_model):
        if(self.fl_round>=self.num_rounds):
            self.done=True
        for id in idl:
            print(time.time()-self.t, f'Sending model of id:{id}, model:{agg_model}')
            ray.get(self.client_models_actor_ref.put.remote(id,{'response':'Hi', 'lr':0.01, 'epoch':5, 'momentum': 0.9, 'done':self.done, 'global_epoch':self.fl_round, 'model':agg_model}))
            ray.get(self.client_status_actor_ref.put.remote(id,1,None))
        print(time.time()-self.t, f"Finished Sending/Updating models")
        return
    
    def start(self, idl):
        # Get client models
        config, models = self.get_models(idl)

        # Aggregate
        agg_model = self.aggregate(config, models)

        # Send Clients, models
        self.send_models(idl, agg_model)    