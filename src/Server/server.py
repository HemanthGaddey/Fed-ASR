from multiprocessing import Process, current_process
from .BaseServer import AbstractServerClass
import time, traceback
import grpc
import sys
import threading
from datetime import datetime
from queue import Queue
sys.path.append('../')
from utils import ClientMessage, ServerMessage
from concurrent import futures

#aggregation strategy
from Server.Aggregation.FedAvg import FedAvg
from Server.Aggregation.FedAvg_New import Fedavg_New
from Server.Aggregation.FedBN import Fedbn_New


import proto.cbsp_pb2 as cbsp_pb2
import proto.cbsp_pb2_grpc as cbsp_pb2_grpc
from utils.CommunicationService import CommunicationService
import torch
from Models import simple_model, vgg16, custom_model, digit_model

# def worker_function(name):
#     print(f"{name} started")
#     time.sleep(5)  # Simulating some work
#     print(f"{name} completed")

flag=0

class CommunicationService(cbsp_pb2_grpc.CommunicationServiceServicer):
    # As per docs, this class is not supposed to have init function, I added this just so I can add the cbsp_pb2
    # as class' internal variable and also ClientManager as internal object, consider removing if unexplained errors persist 
    def __init__(self, cbsp_pb2, num_clients, lock, min_clients=None, preserve=None, strategy=None, failed_clients_info=None):
        self.fl_round=0
        self.lock=lock
        #
        self.preserve = preserve
        self.client_models={}
        for i in range(num_clients):
            self.client_models[i]=[]
        self.min_clients=min_clients
        self.agg_client_models={}
        for i in range(num_clients):
            self.agg_client_models[i]=[0, None]
        self.failed_clients_info = failed_clients_info
        self.failed_clients=[]
        self.strategy= strategy
        # self.entering_time = []
        # self.exiting_time = []
        # self.timeout = float('inf')
        # self.intervals=[]
        #
        self.CbspMsg = cbsp_pb2
        self.cm = ClientMessage.ClientMessage(cbsp_pb2)
        self.sm = ServerMessage.ServerMessage(cbsp_pb2)

        # -------------------------select the model here----------------------
        # self.model = custom_model.CNN()
        # self.model = simple_model.SimpleModel()
        self.model = digit_model.DigitModel()
        # self.model = vgg16.VGG16(10)
        # -------------------------------------------------------------------
        self.glob_params = self.model.parameters()
        self.clients=[]
        # self.queue=Queue(maxsize=num_clients)

        #shared memory
        self.queue=[]
        self.waiting_list = [False] * num_clients
        
    def BidirectionalStream(self, request_iterator, context):        
        # print(hasattr(request_iterator,'send_results')) # Warning: Doesn't work - A grpc msg has attributes for all msg types, just that rest of them are empty
        # Check the type of message received and process accordingly
        if(request_iterator.WhichOneof("client_message") == "get_parameters"):

            # print("Received GetParameters message")
            info = self.cm.deserializeGetParametersMsg(request_iterator)
            server_response = self.sm.serializeSendParametersMsg({'response':'Hi', 'lr':0.01, 'epoch':5, 'momentum': 0.9}, self.model)
            # print("Server sending:", info, self.model)

        elif(request_iterator.WhichOneof("client_message") == "get_config"):
            info = self.cm.deserializeGetConfigMsg(request_iterator)
            print(f"{datetime.now()}Server received GetConfig message from Client",info['id'] )            
            done = False
            if(self.fl_round > 450):
                done = True
            # server_response = self.sm.serializeSendParametersMsg({'response':'Hi', 'lr':0.001, 'epoch':5, 'momentum': 0.9, 'done':done}, self.model)
            try:
                # print("##########################################################",info['id'],self.agg_client_models[info['id']])
                while True:
                    if self.waiting_list[info['id']] == False:
                        if self.agg_client_models[info['id']][0]==1:
                            self.agg_client_models[info['id']][0]=0
                            if(self.strategy == 'fedavg'):
                                server_response = self.sm.serializeSendParametersMsg({'response':'Hi', 'lr':0.01, 'epoch':5, 'momentum': 0.9, 'done':done, 'global_epoch':self.fl_round}, self.agg_client_models[info['id']][1])
                            elif(self.strategy == 'fedbn'):
                                server_response = self.sm.serializeSendParametersMsg({'response':'Hi', 'lr':0.01, 'epoch':5, 'momentum': 0.9, 'done':done, 'global_epoch':self.fl_round}, self.agg_client_models[info['id']])
                            break
            except Exception as e:
                print('Error:',e)
                traceback_info = traceback.format_exc()
                print(f"\033[31m TRACEBACK:{traceback_info}\033[0m",)
        
        elif(request_iterator.WhichOneof("client_message") == "send_parameters"):
            try:
                info, params, model = self.cm.deserializeSendParametersMsg(request_iterator, self.model)
            except Exception as e:
                print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<",e,"---------------------------------")
            info['id'] = info['id'].pop()
            
            print(f"\033[92m {datetime.now()} Server: Client {info['id']} sent trained model.\033[0m")
            if len(self.client_models[info['id']])==self.preserve:
                with self.lock:
                    self.client_models[info['id']].pop(0)
                    self.client_models[info['id']].append(model)
            else:
                with self.lock:
                    self.client_models[info['id']].append(model)
            
            with self.lock:
                self.queue.append([info['id'],model])
                self.waiting_list[info['id']] = True

            print(f"\033[92m {datetime.now()} Server: Client {info['id']} status: {len(self.queue)} in queue: {[i[0] for i in self.queue]}.\033[0m")
                
            # Aggregation Code
            server_response = self.sm.serializeNormalResponseMsg({'_response','parameters are taken'},f"Client: {info['id']} will receive aggregated parameters now.")
            while(self.waiting_list[info['id']] == True):
                print(self.waiting_list)
                time.sleep(6)
                print(f" {datetime.now()} Server: Client {info['id']} status: {len(self.queue)} in queue: {[i[0] for i in self.queue]}.")
                if len(self.queue)>=self.min_clients:
                    selected_models = []
                    #new
                    selected_model_ids=[]
                    #new
                    s_m={} #-->dict: {'id':'model'}
                    for i in range(self.min_clients):
                        with self.lock:
                            x=self.queue.pop(0)
                            # self.waiting_list[x[0]] = False
                            selected_models.append(x[1])
                            selected_model_ids.append(x[0])
                            s_m[x[0]]=x[1]
                        print(f"!!!!!!!!!!!!!!!!!!!!! Client{x[0]} is getting aggregated !!!!!!!!!!!!!!!!!!!!!")
                    
                    try:
                        if self.strategy == 'fedavg':
                            with self.lock:
                                self.agg_client_models=Fedavg_New(selected_models, self.model, self.agg_client_models, selected_model_ids)
                            print(f"\033[34m {datetime.now()} self.model is updated \033[0m")
                        elif self.strategy == "fedbn":
                            num = len(selected_models)
                            with self.lock:
                                self.model, self.agg_client_models = Fedbn_New(self.model, selected_models, [1/num]*num)
                        else:
                            # self.model, self.agg_client_models  = FedASR(selected_models, self.model, self.failed_clients, self.min_clients, self.client_models, [1/num]*num)
                            pass

                        print("===========================================after aggregation",selected_model_ids,self.waiting_list)
                    except Exception as e:
                        print(f"\033[31m Failure while aggregating\033[0m")
                        error = traceback.print_exc()
                        print(f"\033[31m ERROR: {e} \033[0m")
                        print(f"\033[31m Traceback: {error} \033[0m")
                    
                    for id in selected_model_ids:
                        with self.lock:
                            self.waiting_list[id]=False

                    with self.lock:
                        self.fl_round+=1
                    # idx=0
                    # for round in self.failed_clients_info:
                    #     if round>0 and round<=self.fl_round :
                    #         with self.lock:
                    #             self.failed_clients.append(idx)
                    #     idx+=1
                    print(f"\033[92m {datetime.now()} CLient {info['id']} FL round: {self.fl_round} completed \033[0m")

                    # server_response = self.sm.serializeNormalResponseMsg({'_response','parameters are taken'},f"Client: {info['id']} will receive aggregated parameters now.")
                    break
           
        elif(request_iterator.WhichOneof("client_message") == "send_results"):
            print("Received SendResults message")
            info, results = self.cm.deserializeSendResultsMsg(request_iterator)
            print(info, results)
        else:
            print("ERROR: Received unknown message type")

        return server_response

class FLServer(AbstractServerClass):
    def __init__(self, name, port, num_clients, min_clients, strategy, failed_clients_info):
        super().__init__(name)
        self.name = name
        self.CbspMsg = cbsp_pb2
        self.cm = ClientMessage.ClientMessage(cbsp_pb2)
        self.sm = ServerMessage.ServerMessage(cbsp_pb2)
        self.failed_clients_info=failed_clients_info
        self.strategy=strategy
        self.preserve=5
        options = [
        ('grpc.max_receive_message_length', 1024 * 1024 * 1000)  # Adjust the size as needed
    ]
        lock = threading.Lock()
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=20), options=options)
        cbsp_pb2_grpc.add_CommunicationServiceServicer_to_server(CommunicationService(cbsp_pb2, num_clients, lock, min_clients, self.preserve, self.strategy, self.failed_clients_info), server)
        
        server.add_insecure_port(f'[::]:{port}')
        server.start()
        print('Server Started...')
        print(f"-------------------------Server Started, Listening on port: {port}-------------------------")
        print(f"------------------------------------exp:{self.name}------------------------------------")
        server.wait_for_termination()
        
    def aggregate(self, model):
        # self.aggregator
        if flag==0:
            flag=1
            return model
        else:
            model =  FedAvg

        
        return 

    def send_params(self):
        
        return

    def evaluate(self):
        
        return 

    def recieve_params(self):
        
        return 

