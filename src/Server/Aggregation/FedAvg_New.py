import numpy as np
from Models.simple_model import SimpleModel
import torch

def running_model_avg(current, next, scale):
    if current == None:
        current = next
        for key in current:
            current[key] = current[key] * scale
    else:
        for key in current:
            current[key] = current[key] + (next[key] * scale)
    return current

def Fedavg_New(models, global_model, agg_client_models, selected_model_ids):
    num_clients=len(models)
    running_avg = None
    for model in models:
        next = model.state_dict()
        running_avg = running_model_avg(running_avg, next, 1/num_clients)
    # global_model_dict = global_model.state_dict()
    # running_avg = running_model_avg(running_avg, global_model_dict, )
    # global_model.load_state_dict(running_avg)

    for id in selected_model_ids:
        agg_client_models[id]=[1,global_model]
    return agg_client_models
