# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

from proto import cbsp_pb2 as proto_dot_cbsp__pb2


class CommunicationServiceStub(object):
    """DictionaryService
    """

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.BidirectionalStream = channel.unary_unary(
                '/cbsp.CommunicationService/BidirectionalStream',
                request_serializer=proto_dot_cbsp__pb2.ClientMessage.SerializeToString,
                response_deserializer=proto_dot_cbsp__pb2.ServerMessage.FromString,
                )


class CommunicationServiceServicer(object):
    """DictionaryService
    """

    def BidirectionalStream(self, request, context):
        """rpc TransmitDictionary(PytorchParameters) returns (Response);
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_CommunicationServiceServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'BidirectionalStream': grpc.unary_unary_rpc_method_handler(
                    servicer.BidirectionalStream,
                    request_deserializer=proto_dot_cbsp__pb2.ClientMessage.FromString,
                    response_serializer=proto_dot_cbsp__pb2.ServerMessage.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'cbsp.CommunicationService', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class CommunicationService(object):
    """DictionaryService
    """

    @staticmethod
    def BidirectionalStream(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/cbsp.CommunicationService/BidirectionalStream',
            proto_dot_cbsp__pb2.ClientMessage.SerializeToString,
            proto_dot_cbsp__pb2.ServerMessage.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
