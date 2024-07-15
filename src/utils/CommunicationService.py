import torch

import proto.cbsp_pb2 as cbsp_pb2
import proto.cbsp_pb2_grpc as cbsp_pb2_grpc
from .ClientMessage import ClientMessage
from .ServerMessage import ServerMessage


class CommunicationService(cbsp_pb2_grpc.CommunicationServiceServicer):
    # As per docs, this class is not supposed to have init function, I added this just so I can add the cbsp_pb2
    # as class' internal variable and also ClientManager as internal object, consider removing if unexplained errors persist 
    def __init__(self, cbsp_pb2):
        self.CbspMsg = cbsp_pb2
        self.cm = ClientMessage(self.CbspMsg)
        self.sm = ServerMessage(self.CbspMsg)
        
    def BidirectionalStream(self, request_iterator, context):        
        # print(hasattr(request_iterator,'send_results')) # Warning: Doesn't work - A grpc msg has attributes for all msg types, just that rest of them are empty
        # Check the type of message received and process accordingly
        match request_iterator.WhichOneof("client_message"):
            case "get_parameters":
                print("Received GetParameters message")
                info = self.cm.deserializeGetParametersMsg(request_iterator)
                print(info)
            case "get_config":
                print("Received GetConfig message")
                info = self.cm.deserializeGetConfigMsg(request_iterator)
                print(info)
            case "send_parameters":
                print("Received SendParameters message")
                model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
                info, params, model = self.cm.deserializeSendParametersMsg(request_iterator, model)
                print(info,len(params),model)
                # recv_param_test(model)
                # info, param_float, param_dtype, model = cm.deserializeSendParametersMsg(request_iterator, model) # Alternate way to get updated model as well
                
            case "send_results":
                print("Received SendResults message")
                info, results = self.cm.deserializeSendResultsMsg(request_iterator)
                print(info, results)
            case _:
                print("ERROR: Received unknown message type")
        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
        server_response = self.sm.serializeSendParametersMsg({'aa':'moddale'}, model)
        return server_response
