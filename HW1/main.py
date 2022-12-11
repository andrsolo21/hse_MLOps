from fastapi import FastAPI, UploadFile, File, HTTPException
from api_models import *
import json

import grpc
import messages_pb2
import messages_pb2_grpc

app = FastAPI()

# Start servers:
# uvicorn main:app --reload
# python server_grpc.py

GRPC_SERVER = '172.19.0.3:50051'


@app.get("/available_model_types/", response_model=AvailableModelTypeRespond)
async def get_available_model_types():
    """
    Get available model types
    :return: AvailableModelTypeRespond
    """
    with grpc.insecure_channel(GRPC_SERVER) as channel:
        stub = messages_pb2_grpc.GreeterStub(channel)
        response = stub.GetAvailableModelTypes(messages_pb2.HelloRequest(name='1'))
    return AvailableModelTypeRespond(available_model_types=list(response.available_model_types))


@app.get("/models_list/", response_model=ModelsListRespond)
async def get_models_list():
    """
    Get list of created models
    :return: ModelsListRespond
    """
    with grpc.insecure_channel(GRPC_SERVER) as channel:
        stub = messages_pb2_grpc.GreeterStub(channel)
        response = stub.GetMLModelsList(messages_pb2.HelloRequest(name='1'))
    return ModelsListRespond(models_list=list(response.ml_models_list))


@app.post("/models/{model_type}/create/",
          status_code=201,
          response_model=CreateModelRespond)
async def create_model(model_type: ModelType,
                       model_params: ONEClassParams | None = None
                       # model_params: RLParams | DTCParams | DTRParams | None = None # fastAPI not supported(
                       ):
    """
    Create model with given parameters
    :param model_type: type of crated model
    :param model_params: parameters for creating data
    :return: CreateModelRespond
    """

    with grpc.insecure_channel(GRPC_SERVER) as channel:
        stub = messages_pb2_grpc.GreeterStub(channel)
        # params = json.dumps(model_params.dict())
        params = json.dumps(reformat_model_params(model_params=ONEClassParams(**model_params.dict()),
                                                  model_type=model_type).dict())
        response = stub.CreateModel(messages_pb2.CreateModelRequest_2(model_type=model_type, model_params=params))
    return CreateModelRespond(path=response.model_path, model_type=response.model_type)


@app.post("/models/{model_name}/fit/",
          response_model=FitModelRespond,
          responses={404: {"model": RespondError}, 415: {"model": RespondError}, 400: {"model": RespondError}})
async def fit_model(model_name: str,
                    target_column: str = "target",
                    uploaded_file: UploadFile = File(description="A file read as UploadFile")
                    ):
    """
    Fit or reFit model with given dataset
    :param model_name: Identifying model name
    :param target_column: target column in dataset
    :param uploaded_file: dataset for train
    :return: FitModelRespond
    """
    with grpc.insecure_channel(GRPC_SERVER) as channel:
        stub = messages_pb2_grpc.GreeterStub(channel)
        response = stub.FitModel(send_file_by_grpc(uploaded_file=uploaded_file,
                                                   model_name=model_name,
                                                   target_column=target_column,
                                                   chunk_size=1024))
        if response.error_code == 0:
            return FitModelRespond(is_trained=response.is_trained)
        else:
            raise HTTPException(status_code=response.error_code, detail=response.error_message)


@app.post("/models/{model_name}/predict/",
          response_model=PredictModelRespond,
          responses={404: {"model": RespondError},
                     415: {"model": RespondError},
                     400: {"model": RespondError},
                     418: {"model": RespondError}})
async def get_predictions(model_name: str,
                          uploaded_file: UploadFile = File(description="A file read as UploadFile")
                          ):
    """
    Predict data for given dataset
    :param model_name: Identifying model name
    :param uploaded_file: Dataset for predicting data
    :return: PredictModelRespond
    """
    with grpc.insecure_channel(GRPC_SERVER) as channel:
        stub = messages_pb2_grpc.GreeterStub(channel)
        response = stub.PredictData(send_file_by_grpc(uploaded_file=uploaded_file,
                                                      model_name=model_name,
                                                      chunk_size=1024))
        if response.error_code == 0:
            return PredictModelRespond(predict=list(response.predict))
        else:
            raise HTTPException(status_code=response.error_code, detail=response.error_message)


@app.get("/models/{model_name}/info/",
         response_model=GetInfoRespond,
         responses={404: {"model": RespondError}})
async def get_model_info(model_name: str):
    """
    Get basic info about model
    :param model_name: Identifying model name
    :return: basic info about model GetInfoRespond
    """
    with grpc.insecure_channel(GRPC_SERVER) as channel:
        stub = messages_pb2_grpc.GreeterStub(channel)
        response = stub.GetModelInfo(messages_pb2.ModelName(model_name=model_name))
        if response.error_code == 0:
            # model_info_resp = {}
            # model_info = json.loads(response.model_params)
            # for key in model_info:
            #     if model_info[key] is not None:
            #         model_info_resp[key] = model_info[key]

            return GetInfoRespond(**json.loads(response.model_params))
        else:
            raise HTTPException(status_code=response.error_code, detail=response.error_message)


@app.delete("/models/{model_name}/delete/")
async def delete_model(model_name: str):
    """
    Deleting model with the given model name
    :param model_name: Identifying model name
    :return: ModelsListRespond
    """
    with grpc.insecure_channel(GRPC_SERVER) as channel:
        stub = messages_pb2_grpc.GreeterStub(channel)
        response = stub.DeleteModel(messages_pb2.ModelName(model_name=model_name))
        if response.error_code == 0:
            return ModelsListRespond(models_list=list(response.ml_models_list))
        else:
            raise HTTPException(status_code=response.error_code, detail=response.error_message)


def send_file_by_grpc(uploaded_file: UploadFile, model_name: str, target_column: str = None, chunk_size: int = 1024):
    """
    Streaming file by grpc
    :param uploaded_file: uploaded file for streaming
    :param model_name: model name
    :param target_column: target column
    :param chunk_size: chunk size
    :return: stream of data
    """
    extension = uploaded_file.filename.split(".")[-1]
    if target_column is None:
        metadata = messages_pb2.PredictMetaData(extension=extension, model_name=model_name)
        yield messages_pb2.SendFile(predict_metadata=metadata)
    else:
        metadata = messages_pb2.FitMetaData(extension=extension, model_name=model_name, target_column=target_column)
        yield messages_pb2.SendFile(fit_metadata=metadata)
    while True:
        chunk = uploaded_file.file.read(chunk_size)
        if chunk:
            entry_request = messages_pb2.SendFile(chunk_data=chunk)
            yield entry_request
        else:
            return
