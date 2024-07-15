import numpy as np
from Models.simple_model import SimpleModel
import torch

def FedBN(models, global_model, client_ids):
    # Initialize an empty list to store aggregated parameters
    aggregated_params = {}
    for _id in client_ids:
        aggregated_params[_id]=[]

    models.append(global_model)
    # Get the total number of models
    num_models = len(models)

    # Get the state dictionary keys of the first model to determine the layer names
    first_model_state_dict = models[0].state_dict()
    layer_names = [key.split('.')[:-1] for key in first_model_state_dict.keys() if 'weight' in key]
    for ln in layer_names:

        # latest code
        name = ".".join(ln)
        layer_dict={}
        types=[key.split('.')[-1] for key in first_model_state_dict.keys() if f'{name}.' in key]
        for _type in types:
            layer_dict[_type]=[]
        if(len(types)>2):
            for model, _id in zip(models, client_ids):
                state_dict=model.state_dict()
                # for _type in types:
                #     layer_dict[_type].append(state_dict[f'{name}.{_type}'])
                for _type in types:
                    aggregated_params[_id].append(state_dict[f'{name}.{_type}'])

        else:
            for model, _id in zip(models, client_ids):
                state_dict=model.state_dict()
                for _type in types:
                    layer_dict[_type].append(state_dict[f'{name}.{_type}'])
            avg_layer_dict={}
            for _type in types:
                avg_layer_dict[_type]=np.mean(layer_dict[_type], axis=0)
            for _type in types:
                for _id in client_ids:
                    aggregated_params[_id].append(torch.tensor(avg_layer_dict[_type]))

    
    # Assign the aggregated parameters to the new model
    aggregated_models={}
    for _id in client_ids:
        # Create a new model to hold the aggregated parameters
        aggregated_model = DigitModel()  # Assuming you have a SimpleModel class defined
    
        layer_index = 0
        for name, param in aggregated_model.named_parameters():
            param.data = torch.tensor(aggregated_params[_id  ][layer_index], dtype=param.data.dtype)
            layer_index += 1
        aggregated_models[_id]=aggregated_model

    print("*************FL aggregation done !*************")

    return aggregated_models


def Fedbn_New(server_model, models, client_weights):
    client_num=len(models)
    with torch.no_grad():
        for key in server_model.state_dict().keys():
            if 'bn' not in key:
                temp = torch.zeros_like(server_model.state_dict()[key], dtype=torch.float32)
                for client_idx in range(client_num):
                    temp += client_weights[client_idx] * models[client_idx].state_dict()[key]
                server_model.state_dict()[key].data.copy_(temp)
                for client_idx in range(client_num):
                    models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])
    return server_model, models
