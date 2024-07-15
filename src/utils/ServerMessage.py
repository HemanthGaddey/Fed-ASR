import pickle
import numpy as np
import grpc
import proto.cbsp_pb2 as cbsp_pb2
import proto.cbsp_pb2_grpc as cbsp_pb2_grpc
import torch

class ServerMessage():
    def __init__(self, CbspMsg):
        self.CbspMsg = CbspMsg
        
    # TODO: Reduce redundancy by separating/modularizing utility functions code
    # Note: All the serialization and deserialization use Lazy Dict serialization!
    # Utility Functions
    def torchModel2NumpyParams(self, model): 
        params = [val.cpu().numpy() for _, val in model.state_dict().items()]
        return params

    def numpyParams2TorchModel(self, model, params): 
        model_params = model.state_dict()
        for key, val in model_params.items():
            if len(params) > 0:
                model_params[key] = torch.tensor(params.pop(0))
        model.load_state_dict(model_params)
        return model
        
    # Orphaned Function
    def serializeDictStrict(self, info): 
        info_ = {}
        for key, value in info.items():
            assert (isinstance(key, str)), "Dictionary Keys contain non String values!"
            assert (isinstance(value, int) or isinstance(value, str) or isinstance(value, float) or isinstance(value, bool)), "Dictionary Contains Unsupported Value types!"
            
            value_ = None
            if(isinstance(value, bool)):    
                value_ = self.CbspMsg.Constant(bool=value)
            elif(isinstance(value, int)):    
                value_ = self.CbspMsg.Constant(sint64=value)
            elif(isinstance(value, float)):    
                value_ = self.CbspMsg.Constant(double=value)
            elif(isinstance(value, str)):    
                value_ = self.CbspMsg.Constant(string=value)
            else:
                print("UNKNOWN ERROR CONVERTING DICT TO GRPC MSG FORMAT :( (chusko)")
                
            info_[key] = value_
        return info_
    
    def serializeDictLazy(self, info):
        return {"dict":self.CbspMsg.Constant(string=str(info))}
        
    def deserializeDictLazy(self, info):
        return eval(info['dict'].string)

    # Msg Serialization Functions        
    def serializePytorchParams(self, params): 
        params_bytelist = []
        for i in params:
            params_bytelist.append(cbsp_pb2.ParameterBytes(tensor=pickle.dumps(i), shape=[0]))
        return cbsp_pb2.PytorchParameters(parameters=params_bytelist, dtype=str(params[0].dtype))
   
    def serializeSendParametersMsg(self, info, model): # TODO: Get Better Naming for these since pytorch specific
        info_=self.serializeDictLazy(info)
        params = self.torchModel2NumpyParams(model)
        params_grpc=self.serializePytorchParams(params) # params_grpc
        return self.CbspMsg.ServerMessage(
            get_parameters = self.CbspMsg.ServerMessage.SendParameters( # TODO: Change this get_parameters to send_parameters
                type=self.CbspMsg.ServerMessage.SEND_PARAMETERS,
                info=info_,
                parameters=params_grpc
            )
        )       

    def serializeSendConfigMsg(self, info):
        info_=self.serializeDictLazy(info)
        return self.CbspMsg.ServerMessage(
            send_config = self.CbspMsg.ServerMessage.SendConfig(
                type=self.CbspMsg.ServerMessage.SEND_CONFIG,
                info=info_
            )
        )
        
    def serializeNormalResponseMsg(self, info, response):
        info_=self.serializeDictLazy(info)
        return self.CbspMsg.ServerMessage(
            normal_response = self.CbspMsg.ServerMessage.NormalResponse(
                type=self.CbspMsg.ServerMessage.MESSAGE_TYPE.NORMAL_RESPONSE,
                info=info_,
                response=response
            )
        )
        
    # Msg Deserialization Functions
    def deserializePytorchParams(self, params_grpc): 
        params_bytelist = params_grpc.parameters
        params_dtype = params_grpc.dtype # Redundant
        params = []
        for i in params_bytelist:
            params.append(pickle.loads(i.tensor))
        return params
        
    def deserializeSendParametersMsg(self, request, model=None):
        info_ = request.get_parameters.info # TODO: Change this get_parameters to send_parameters
        param_bytes = request.get_parameters.parameters # TODO: Change this get_parameters to send_parameters
        
        info = self.deserializeDictLazy(info_)
        params = self.deserializePytorchParams(param_bytes)

        if(model):
            model = self.numpyParams2TorchModel(model, params)

        return info, params, model # dtype process is sus

    def deserializeSendConfigMsg(self, request):
        info_ = request.send_config.info
        info = self.deserializeDictLazy(info_)
        return info
        
    def deserializeNormalResponseMsg(self, request):
        info_ = request.normal_response.info
        info = self.deserializeDictLazy(info_)
        response = request.normal_response.response
        return info, response