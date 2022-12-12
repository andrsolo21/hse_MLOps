from fastapi import HTTPException
from concurrent import futures
import logging
import json
import grpc
import messages_pb2
import messages_pb2_grpc

from ml_models import ML_MODELS, MLModel, convert_byte_data
from work_with_database import DBModel


class Greeter(messages_pb2_grpc.GreeterServicer):

    def GetAvailableModelTypes(self, request, context):
        """
        GetAvailableModelTypes
        :param request: request
        :param context: context
        :return: AvailableModelTypes
        """
        logging.getLogger(__name__).log(level=2, msg="Send AvailableModelTypes")
        return messages_pb2.AvailableModelTypes(available_model_types=list(ML_MODELS.keys()))

    def GetMLModelsList(self, request, context):
        """
        GetMLModelsList
        :param request: request
        :param context: context
        :return: MLModelsList
        """
        logging.getLogger(__name__).log(level=3, msg="Send MLModelsList")
        return messages_pb2.MLModelsList(ml_models_list=DBModel.get_models_list())

    def CreateModel(self, request, context):
        """
        CreateModel
        :param request: request
        :param context: context
        :return: ResponseCreatedModel
        """

        logging.getLogger(__name__).log(level=2, msg="Send CreateModel")
        model_type = request.model_type
        model_params = json.loads(request.model_params)

        if model_params is None:
            params = {}
        else:
            params = model_params

        ml_model = MLModel(type_model=model_type, params=params)
        ml_model.dump_model()
        return messages_pb2.ResponseCreatedModel(model_name=ml_model.model_name)

    def FitModel(self, request_iterator, context):
        """
        FitModel
        :param request_iterator: iterator for data from request
        :param context: context
        :return: ResponseCreatedModel
        """
        logging.getLogger(__name__).log(level=2, msg="Send FitModel")
        data = bytearray()
        extension = None
        model_name = None
        target_column = None

        for request in request_iterator:
            if request.fit_metadata.extension and request.fit_metadata.model_name and \
                    request.fit_metadata.target_column:
                extension = request.fit_metadata.extension
                model_name = request.fit_metadata.model_name
                target_column = request.fit_metadata.target_column
                continue
            data.extend(request.chunk_data)


        try:
            data = convert_byte_data(data, extension)
            ml_model = MLModel(model_name=model_name)
            ml_model.fit(data, target_column)
            ml_model.dump_model()
        except HTTPException as exc:
            return messages_pb2.ResponseFitModel(is_trained=False,
                                                 error_code=exc.status_code,
                                                 error_message=exc.detail)
        return messages_pb2.ResponseFitModel(is_trained=ml_model.is_trained,
                                             error_code=0,
                                             error_message=None)

    def PredictData(self, request_iterator, context):
        """
        PredictData
        :param request_iterator: iterator for data from request
        :param context: context
        :return: ResponsePredictData
        """
        logging.getLogger(__name__).log(level=2, msg="Send PredictData")
        data = bytearray()
        extension = None
        model_name = None

        for request in request_iterator:
            if request.predict_metadata.extension and request.predict_metadata.model_name:
                extension = request.predict_metadata.extension
                model_name = request.predict_metadata.model_name
                continue
            data.extend(request.chunk_data)

        try:
            data = convert_byte_data(data, extension)
            ml_model = MLModel(model_name=model_name)
            result = ml_model.predict(data)
        except HTTPException as exc:
            return messages_pb2.ResponsePredictData(predict=None,
                                                    error_code=exc.status_code,
                                                    error_message=exc.detail)

        return messages_pb2.ResponsePredictData(predict=list(result),
                                                error_code=0,
                                                error_message=None)

    def GetModelInfo(self, request, context):
        """
        GetModelInfo
        :param request: request
        :param context: context
        :return: MLModelsList
        """
        logging.getLogger(__name__).log(level=2, msg="Send GetModelInfo")
        try:
            ml_model = MLModel(model_name=request.model_name)
            data = ml_model.get_info()
        except HTTPException as exc:
            return messages_pb2.ResponseModelInfo(model_params=None,
                                                  error_code=exc.status_code,
                                                  error_message=exc.detail)
        return messages_pb2.ResponseModelInfo(model_params=json.dumps(data),
                                              error_code=0,
                                              error_message=None)

    def DeleteModel(self, request, context):
        """
        DeleteModel
        :param request: request
        :param context: context
        :return: MLModelsList
        """
        logging.getLogger(__name__).log(level=2, msg="Send DeleteModel")
        try:
            ml_model = MLModel(model_name=request.model_name)
            ml_model.delete_model()
            models_list = DBModel.get_models_list()
        except HTTPException as exc:
            return messages_pb2.MLModelsList(ml_models_list=None,
                                             error_code=exc.status_code,
                                             error_message=exc.detail)
        return messages_pb2.MLModelsList(ml_models_list=models_list,
                                         error_code=0,
                                         error_message=None)


def serve():
    """
    Start grpc server
    :return: None
    """
    DBModel.create_table_if_not_exists()

    port = '50051'
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    messages_pb2_grpc.add_GreeterServicer_to_server(Greeter(), server)
    server.add_insecure_port('[::]:' + port)
    server.start()
    logging.getLogger(__name__).log(level=2, msg="Server started, listening on " + port)
    server.wait_for_termination()


if __name__ == '__main__':
    # logging.basicConfig()
    # logging.getLogger(__name__).setLevel(logging.INFO)
    serve()
