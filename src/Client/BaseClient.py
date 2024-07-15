import torch
from abc import ABC
from typing import List

class AbstractClientClass(ABC):

    def __init__(self, id):
        

        super().__init__()

        self.id  = id

    def send_params(self, model):
        '''
        the parameters will be shared
        '''
        return []

    def train(self, parameters: List[float], config: dict):
        '''
        this starts training and returns parameters, if needed other info can also be shared
        '''
        _=(self, parameters, config)
        return []

    def evaluate(self, parameters, config):
        '''
        this evaluates the aggregated parameters with its locally available data and
        returns the metrics, if needed other info can also be shared
        '''
        _=(self, parameters, config)

        return []
    
