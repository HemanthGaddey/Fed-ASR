import numpy as np
import grpc
import proto.cbsp_pb2 as cbsp_pb2
import proto.cbsp_pb2_grpc as cbsp_pb2_grpc
from concurrent import futures

class CommunicationService(cbsp_pb2_grpc.CommunicationServiceServicer):
    def BidirectionalStream(self, request_iterator, context):
        print('got the message')
        # print(request_iterator)
        # for client_message in request_iterator:
        #     print('got the message')
        #     print(client_message)
            
        server_response = cbsp_pb2.ServerMessage(
            normal_response = cbsp_pb2.ServerMessage.NormalResponse(
                type=cbsp_pb2.ServerMessage.MESSAGE_TYPE.NORMAL_RESPONSE,
                info={'msg':cbsp_pb2.Constant(string='aa vinnanule')},
                response="OK"
            )
        )

        # Case for non stream messages
        match request_iterator:
            case 1:
                pass
        
        #print(request_iterator)
        param_bytes = request_iterator.send_parameters.parameters.parameters
        param_dtype = request_iterator.send_parameters.parameters.dtype
        param_float = []
        for i in param_bytes:
            param_float.append(np.frombuffer(i.tensor, dtype=param_dtype).reshape(i.shape))
        
        print(param_float)
        return server_response


def run_server():
    options = [
        ('grpc.max_receive_message_length', 1024 * 1024 * 1000)  # Adjust the size as needed
    ]
    server = grpc.server(futures.ThreadPoolExecutor(), options=options)
    cbsp_pb2_grpc.add_CommunicationServiceServicer_to_server(CommunicationService(), server)
    
    server.add_insecure_port('[::]:50051')
    server.start()
    print('Server Started')
    server.wait_for_termination()

if __name__ == '__main__':
    run_server()