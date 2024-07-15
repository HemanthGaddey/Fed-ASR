import pickle 
import numpy as np
import grpc
import proto.cbsp_pb2 as cbsp_pb2
import proto.cbsp_pb2_grpc as cbsp_pb2_grpc
import torch

class ClientMessage():
    def __init__(self, CbspMsg):
        self.CbspMsg = CbspMsg
    
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
                      
    def serializeGetParametersMsg(self, info):
        info_=self.serializeDictLazy(info)
        return self.CbspMsg.ClientMessage(
            get_parameters = self.CbspMsg.ClientMessage.GetParameters(
                type=self.CbspMsg.ClientMessage.GET_PARAMETERS,
                info=info_
            )
        )
        
    def serializeGetConfigMsg(self, info):
        info_=self.serializeDictLazy(info)
        return self.CbspMsg.ClientMessage(
            get_config = self.CbspMsg.ClientMessage.GetConfig(
                type=self.CbspMsg.ClientMessage.GET_CONFIG,
                info=info_
            )
        )
        
    def serializeSendParametersMsg(self, info, model):
        info_=self.serializeDictLazy(info)
        params = self.torchModel2NumpyParams(model)
        params_grpc=self.serializePytorchParams(params) # params_grpc
        return self.CbspMsg.ClientMessage(
            send_parameters = self.CbspMsg.ClientMessage.SendParameters(
                type=self.CbspMsg.ClientMessage.SEND_PARAMETERS,
                info=info_,
                parameters=params_grpc
            )
        )

    def serializeSendResultsMsg(self, info, results):
        info_=self.serializeDictLazy(info)
        results_=self.serializeDictLazy(results)
        return self.CbspMsg.ClientMessage(
            send_results = self.CbspMsg.ClientMessage.SendResults(
                type=self.CbspMsg.ClientMessage.SEND_RESULTS,
                info=info_,
                results=results_
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
        
    def deserializeGetParametersMsg(self, request):
        info_ = request.get_parameters.info

        info = self.deserializeDictLazy(info_)

        return info
        
    def deserializeGetConfigMsg(self, request):
        info_ = request.get_config.info

        info = self.deserializeDictLazy(info_)

        return info
        
    def deserializeSendParametersMsg(self, request, model=None):
        info_ = request.send_parameters.info
        params_grpc = request.send_parameters.parameters
        
        info = self.deserializeDictLazy(info_)
        params = self.deserializePytorchParams(params_grpc)

        if(model):
            model = self.numpyParams2TorchModel(model, params)

        return info, params, model # dtype process is sus

    def deserializeSendResultsMsg(self, request):
        info_ = request.send_results.info
        results_ = request.send_results.results

        info = self.deserializeDictLazy(info_)
        results = self.deserializeDictLazy(results_)

        return info, results