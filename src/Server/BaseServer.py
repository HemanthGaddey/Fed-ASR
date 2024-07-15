import torch
from abc import ABC, abstractmethod
from queue import Queue

class AbstractServerClass(ABC):
    """This is an abstract base class for FL Server"""
    def __init__(self, name : str):
        self.name  = name

    @abstractmethod  
    def aggregate(self):
        # self.aggregator
        '''the aggregated weights should be sent to all the clients using the 'clients' list'''
        return

    @abstractmethod
    def send_params(self):
        """This function sends the aggregated weights to clients"""
        return

    @abstractmethod
    def evaluate(self):
        """This function gets the evaluations of selected clients and aggregates the results"""
        return

    @abstractmethod
    def recieve_params(self):
        """This function retreives the weights of clients in the ClientSelectionQueue"""

        return

# mode : str,
# status : bool,
# aggregation_func : str,
# stop_event: None,
# time: float,
# min_clients: int,
# results: Queue,
# clients: list,