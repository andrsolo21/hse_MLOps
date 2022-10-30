from fastapi import FastAPI, UploadFile, File, HTTPException
from ml_models import ML_MODELS, get_ml_models_list, MLModel
from api_models import *
import pandas as pd

app = FastAPI()

# Start server:
# uvicorn main:app --reload


@app.get("/available_model_types/", response_model=AvailableModelTypeRespond)
async def get_available_model_types():
    """
    Get available model types
    :return: AvailableModelTypeRespond
    """
    return AvailableModelTypeRespond(available_model_types=list(ML_MODELS.keys()))


@app.get("/models_list/", response_model=ModelsListRespond)
async def get_models_list():
    """
    Get list of created models
    :return: ModelsListRespond
    """
    models_list, _ = get_ml_models_list()
    return ModelsListRespond(models_list=list(models_list))


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
    model_params = reformat_model_params(model_params, model_type)

    print(model_params)
    if model_params is None:
        params = {}
    else:
        params = model_params.dict()

    ml_model = MLModel(type_model=model_type, params=params)
    model_path = ml_model.dump_model()
    return CreateModelRespond(path=model_path, model_params=model_params, model_type=model_type)


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
    data = convert_uploaded_file(uploaded_file)
    ml_model = MLModel(model_name=model_name)
    ml_model.fit(data, target_column)
    ml_model.dump_model()
    return FitModelRespond(is_trained=ml_model.is_trained)


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
    data = convert_uploaded_file(uploaded_file)
    ml_model = MLModel(model_name=model_name)
    result = ml_model.predict(data)
    return PredictModelRespond(predict=list(result))


@app.get("/models/{model_name}/info/",
         response_model=GetInfoRespond,
         responses={404: {"model": RespondError}})
async def get_model_info(model_name: str):
    """
    Get basic info about model
    :param model_name: Identifying model name
    :return: basic info about model GetInfoRespond
    """
    ml_model = MLModel(model_name=model_name)
    data = ml_model.get_info()
    return GetInfoRespond(**data)


@app.delete("/models/{model_name}/delete/")
async def delete_model(model_name: str):
    """
    Deleting model with the given model name
    :param model_name: Identifying model name
    :return: ModelsListRespond
    """
    ml_model = MLModel(model_name=model_name)
    ml_model.delete_model()
    models_list, _ = get_ml_models_list()
    return ModelsListRespond(models_list=list(models_list))


def convert_uploaded_file(uploaded_file: UploadFile) -> pd.DataFrame:
    """
    Check and convert uploaded file
    :param uploaded_file:
    :return: uploaded dataset
    :raise: HTTPException - "Invalid file type"
    """
    if uploaded_file.filename.split(".")[-1] == "xlsx" or uploaded_file.filename.split(".")[-1] == "xls":
        data = pd.read_csv(uploaded_file.file)
    elif uploaded_file.filename.split(".")[-1] == "csv":
        data = pd.read_csv(uploaded_file.file)
    else:
        raise HTTPException(status_code=415, detail="Invalid file type")
    return data
