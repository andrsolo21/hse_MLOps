"""The Python implementation of the GRPC helloworld.Greeter client."""

from __future__ import print_function

import logging

import grpc
import messages_pb2
import messages_pb2_grpc


def run():
    # NOTE(gRPC Python Team): .close() is possible on a channel and should be
    # used in circumstances in which the with statement does not fit the needs
    # of the code.
    with grpc.insecure_channel('localhost:50051') as channel:
        stub = messages_pb2_grpc.GreeterStub(channel)
        response = stub.GetAvailableModelTypes(messages_pb2.HelloRequest(name='1'))
    print(response.available_model_types)
    print(type(list(response.available_model_types)))


if __name__ == '__main__':
    logging.basicConfig()
    run()
