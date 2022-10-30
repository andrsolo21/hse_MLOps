from fastapi import FastAPI, UploadFile, File, HTTPException
from ml_models import ML_MODELS, get_models_list, MLModel
from api_models import *
from typing import Union
import pandas as pd

app = FastAPI()


# Start server:
# uvicorn main:app --reload


@app.get("/available_model_types/")
async def get_available_model_types():
    """
    Get available model types
    :return:
    """
    return {"available_model_types": list(ML_MODELS.keys())}


@app.get("/models_list/")
async def get_models_list():
    models_list, _ = get_models_list()
    return {"models_list": models_list}


@app.post("/models/{model_type}/create/", status_code=201)
async def create_model(model_type: ModelType,
                       model_params: Union[RLParams, DTCParams, DTRParams] = None
                       # model_params: UnionRLParams | DTCParams | DTRParams | None = None
                       ):
    print(model_params)
    if model_params is None:
        params = {}
    else:
        params = model_params.dict()
    print(type(model_params))

    ml_model = MLModel(type_model=model_type, params=params)
    model_path = ml_model.dump_model()
    return {"model_path": model_path}


@app.post("/models/{model_name}/fit/")
async def fit_model(model_name: str,
                    target_column: str = "target",
                    uploaded_file: UploadFile = File(description="A file read as UploadFile")
                    ):
    data = convert_uploaded_file(uploaded_file)
    ml_model = MLModel(model_name=model_name)
    ml_model.fit(data, target_column)
    ml_model.dump_model()
    return {"is_trained": ml_model.is_trained}


@app.post("/models/{model_name}/predict/")
async def get_predictions(model_name: str,
                          uploaded_file: UploadFile = File(description="A file read as UploadFile")
                          ):
    data = convert_uploaded_file(uploaded_file)
    ml_model = MLModel(model_name=model_name)
    result = ml_model.predict(data)
    return {"predict": list(result)}


@app.get("/models/{model_name}/info/")
async def get_model_info(model_name: str):
    """
    Get basic info about model
    :param model_name: Identifying model name
    :return: basic info about model
    """
    ml_model = MLModel(model_name=model_name)
    return ml_model.get_info()


@app.delete("/models/{model_name}/delete/")
async def delete_model(model_name: str):
    """
    Deleting model with the given model name
    :param model_name: Identifying model name
    :return:
    """
    ml_model = MLModel(model_name=model_name)
    ml_model.delete_model()


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
