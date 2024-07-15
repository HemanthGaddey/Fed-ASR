import numpy as np
from Models.simple_model import SimpleModel
import torch

def FedAvg(models, global_model):
    # Initialize an empty list to store aggregated parameters
    aggregated_params = []
    models.append(globl_model)
    # Get the total number of models
    num_models = len(models)

    # Get the state dictionary keys of the first model to determine the layer names
    first_model_state_dict = models[0].state_dict()
    layer_names = [key.split('.')[0] for key in first_model_state_dict.keys() if 'weight' in key]

    # Loop through each layer of the model
    for layer_name in layer_names:
        # Initialize lists to store weights and biases from all models
        weight_params = []
        bias_params = []

        # Loop through each model to collect weights and biases
        for model in models:
            state_dict = model.state_dict()
            weight_params.append(state_dict[f'{layer_name}.weight'])
            bias_params.append(state_dict[f'{layer_name}.bias'])

        # Compute the average of weights and biases across all models
        averaged_weight_params = np.mean(weight_params, axis=0)
        averaged_bias_params = np.mean(bias_params, axis=0)

        # Append the averaged weight and bias parameters to the aggregated_params list
        aggregated_params.append(torch.tensor(averaged_weight_params))
        aggregated_params.append(torch.tensor(averaged_bias_params))

    # Create a new model to hold the aggregated parameters
    aggregated_model = SimpleModel()  # Assuming you have a SimpleModel class defined

    # Assign the aggregated parameters to the new model
    layer_index = 0
    for name, param in aggregated_model.named_parameters():
        param.data = aggregated_params[layer_index]
        layer_index += 1

    print("*************FL aggregation done !*************")

    return aggregated_model

# Example usage:
# aggregated_model = federated_averaging(list_of_models)
