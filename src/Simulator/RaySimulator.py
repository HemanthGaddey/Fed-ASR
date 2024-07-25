import time
import numpy as np
import ray
import sys
from datetime import datetime
from queue import Queue

sys.path.append('../')

from Client.RayLocalClient import RayLocalClient
from Server.RayLocalServer import RayLocalServer
from RayActors.ClientModelsActor import Client_models_actor
from RayActors.ClientStatusActor import Client_status_actor
from Client.Client import FLClient
from Server.server import FLServer

ray.init(ignore_reinit_error=True)

num_clients = 10
min_agg_clients = 3

client_status_actor_ref = Client_status_actor.remote(num_clients,min_agg_clients)
client_models_actor_ref = Client_models_actor.remote(num_clients,min_agg_clients)

server = FLServer.remote(num_clients, min_agg_clients, client_status_actor_ref, client_models_actor_ref, num_rounds=5)
clients = [FLClient.remote(i,f'Client {i}', dataset_dir, server, client_status_actor_ref,client_models_actor_ref, fail_round=-1, device='cuda') for i in range(num_clients)]

ray.wait([*[client.loop.remote() for client in clients]]) #server.start.remote([]),