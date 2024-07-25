import time
import numpy as np
import ray

@ray.remote
class Client_status_actor:
    def __init__(self,num_clients,min_clients):
        self.client_status = [1]*num_clients
        self.num_clients = num_clients
        self.min_clients = min_clients
        
        self.t = time.time()
        print(self.t,'CLient Status actor initialized')
#         self.server_actor_ref = server_actor_ref
    
    def get(self,index):
        print(time.time()-self.t, f'get, index={index}')
        return self.client_status[index]
    
    def get_all(self):
        print(time.time()-self.t, f'get_all')
        return self.client_status
    
    def put(self,index,status_value, server_actor_ref):
        print(time.time()-self.t, f'put, index={index} status_value={status_value}')
        self.client_status[index] = status_value
        
        # Check for aggregation
        if(status_value == 2):
            c=0
            idl = []
            for i in range(len(self.client_status)):
                if(self.client_status[i]==2): 
                    c+=1
                    idl.append(i)
            print(time.time()-self.t, f'In put, c={c} idl={idl} client_status={self.client_status}') 
            if(c>=self.min_clients):
                for i in idl:
                    self.client_status[i]=0
                print(time.time()-self.t, f'In put after min agg clients, c={c} idl={idl} client_status={self.client_status}') 
                server_actor_ref.start.remote(idl)