"""The Python implementation of the GRPC helloworld.Greeter server."""

from concurrent import futures
import logging

import grpc
import messages_pb2
import messages_pb2_grpc

from ml_models import ML_MODELS, get_ml_models_list, MLModel
from api_models import *


class Greeter(messages_pb2_grpc.GreeterServicer):

    def GetAvailableModelTypes(self, request, context):
        """
        GetAvailableModelTypes
        :param request: request
        :param context: context
        :return: AvailableModelTypes
        """
        print("Send AvailableModelTypes")
        return messages_pb2.AvailableModelTypes(available_model_types=list(ML_MODELS.keys()))

    def GetMLModelsList(self, request, context):
        """
        GetMLModelsList
        :param request: request
        :param context: context
        :return: MLModelsList
        """
        print("Send MLModelsList")
        models_list, _ = get_ml_models_list()
        return messages_pb2.MLModelsList(ml_models_list=list(models_list))

    def CreateModel(self, request, context):
        """
        CreateModel
        :param request: request
        :param context: context
        :return: ResponseCreatedModel
        """
        print("start create model")
        models_list, _ = get_ml_models_list()
        model_type = request.model_type
        params = {}
        print("start patch")
        model_params = reformat_model_params(model_params=ONEClassParams(**dict(request)),
                                             model_type=model_type)

        print("stop patch")
        print(model_params)
        if model_params is None:
            params = {}
        else:
            params = model_params.dict()

        ml_model = MLModel(type_model=model_type, params=params)
        model_path = ml_model.dump_model()
        return messages_pb2.ResponseCreatedModel(path=model_path, model_type=model_type)


def serve():
    """
    Start grpc server
    :return: None
    """
    port = '50051'
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    messages_pb2_grpc.add_GreeterServicer_to_server(Greeter(), server)
    server.add_insecure_port('[::]:' + port)
    server.start()
    print("Server started, listening on " + port)
    server.wait_for_termination()


if __name__ == '__main__':
    logging.basicConfig()
    serve()
