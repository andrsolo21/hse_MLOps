# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

import messages_pb2 as messages__pb2


class GreeterStub(object):
    """python -m grpc_tools.protoc -I./grpc --python_out=. --grpc_python_out=. ./grpc/messages.proto

    The greeting service definition.
    The greeting service definition.
    """

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.GetAvailableModelTypes = channel.unary_unary(
                '/helloworld.Greeter/GetAvailableModelTypes',
                request_serializer=messages__pb2.HelloRequest.SerializeToString,
                response_deserializer=messages__pb2.AvailableModelTypes.FromString,
                )
        self.GetMLModelsList = channel.unary_unary(
                '/helloworld.Greeter/GetMLModelsList',
                request_serializer=messages__pb2.HelloRequest.SerializeToString,
                response_deserializer=messages__pb2.MLModelsList.FromString,
                )
        self.CreateModel = channel.unary_unary(
                '/helloworld.Greeter/CreateModel',
                request_serializer=messages__pb2.CreateModelRequest_2.SerializeToString,
                response_deserializer=messages__pb2.ResponseCreatedModel.FromString,
                )
        self.FitModel = channel.stream_unary(
                '/helloworld.Greeter/FitModel',
                request_serializer=messages__pb2.SendFile.SerializeToString,
                response_deserializer=messages__pb2.ResponseFitModel.FromString,
                )
        self.PredictData = channel.stream_unary(
                '/helloworld.Greeter/PredictData',
                request_serializer=messages__pb2.SendFile.SerializeToString,
                response_deserializer=messages__pb2.ResponsePredictData.FromString,
                )
        self.GetModelInfo = channel.unary_unary(
                '/helloworld.Greeter/GetModelInfo',
                request_serializer=messages__pb2.ModelName.SerializeToString,
                response_deserializer=messages__pb2.ResponseModelInfo.FromString,
                )
        self.DeleteModel = channel.unary_unary(
                '/helloworld.Greeter/DeleteModel',
                request_serializer=messages__pb2.ModelName.SerializeToString,
                response_deserializer=messages__pb2.MLModelsList.FromString,
                )


class GreeterServicer(object):
    """python -m grpc_tools.protoc -I./grpc --python_out=. --grpc_python_out=. ./grpc/messages.proto

    The greeting service definition.
    The greeting service definition.
    """

    def GetAvailableModelTypes(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetMLModelsList(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def CreateModel(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def FitModel(self, request_iterator, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def PredictData(self, request_iterator, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetModelInfo(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def DeleteModel(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_GreeterServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'GetAvailableModelTypes': grpc.unary_unary_rpc_method_handler(
                    servicer.GetAvailableModelTypes,
                    request_deserializer=messages__pb2.HelloRequest.FromString,
                    response_serializer=messages__pb2.AvailableModelTypes.SerializeToString,
            ),
            'GetMLModelsList': grpc.unary_unary_rpc_method_handler(
                    servicer.GetMLModelsList,
                    request_deserializer=messages__pb2.HelloRequest.FromString,
                    response_serializer=messages__pb2.MLModelsList.SerializeToString,
            ),
            'CreateModel': grpc.unary_unary_rpc_method_handler(
                    servicer.CreateModel,
                    request_deserializer=messages__pb2.CreateModelRequest_2.FromString,
                    response_serializer=messages__pb2.ResponseCreatedModel.SerializeToString,
            ),
            'FitModel': grpc.stream_unary_rpc_method_handler(
                    servicer.FitModel,
                    request_deserializer=messages__pb2.SendFile.FromString,
                    response_serializer=messages__pb2.ResponseFitModel.SerializeToString,
            ),
            'PredictData': grpc.stream_unary_rpc_method_handler(
                    servicer.PredictData,
                    request_deserializer=messages__pb2.SendFile.FromString,
                    response_serializer=messages__pb2.ResponsePredictData.SerializeToString,
            ),
            'GetModelInfo': grpc.unary_unary_rpc_method_handler(
                    servicer.GetModelInfo,
                    request_deserializer=messages__pb2.ModelName.FromString,
                    response_serializer=messages__pb2.ResponseModelInfo.SerializeToString,
            ),
            'DeleteModel': grpc.unary_unary_rpc_method_handler(
                    servicer.DeleteModel,
                    request_deserializer=messages__pb2.ModelName.FromString,
                    response_serializer=messages__pb2.MLModelsList.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'helloworld.Greeter', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class Greeter(object):
    """python -m grpc_tools.protoc -I./grpc --python_out=. --grpc_python_out=. ./grpc/messages.proto

    The greeting service definition.
    The greeting service definition.
    """

    @staticmethod
    def GetAvailableModelTypes(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/helloworld.Greeter/GetAvailableModelTypes',
            messages__pb2.HelloRequest.SerializeToString,
            messages__pb2.AvailableModelTypes.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def GetMLModelsList(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/helloworld.Greeter/GetMLModelsList',
            messages__pb2.HelloRequest.SerializeToString,
            messages__pb2.MLModelsList.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def CreateModel(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/helloworld.Greeter/CreateModel',
            messages__pb2.CreateModelRequest_2.SerializeToString,
            messages__pb2.ResponseCreatedModel.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def FitModel(request_iterator,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.stream_unary(request_iterator, target, '/helloworld.Greeter/FitModel',
            messages__pb2.SendFile.SerializeToString,
            messages__pb2.ResponseFitModel.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def PredictData(request_iterator,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.stream_unary(request_iterator, target, '/helloworld.Greeter/PredictData',
            messages__pb2.SendFile.SerializeToString,
            messages__pb2.ResponsePredictData.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def GetModelInfo(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/helloworld.Greeter/GetModelInfo',
            messages__pb2.ModelName.SerializeToString,
            messages__pb2.ResponseModelInfo.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def DeleteModel(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/helloworld.Greeter/DeleteModel',
            messages__pb2.ModelName.SerializeToString,
            messages__pb2.MLModelsList.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
