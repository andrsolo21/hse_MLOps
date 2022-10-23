from fastapi import FastAPI, UploadFile, File
from ml_models import ML_MODELS, get_models_list, MLModel
from enum_models import *
from api_models import *
import pandas as pd

app = FastAPI()

# uvicorn main: app - -reload


@app.get("/available_model_types")
async def root():
    return {"available_model_types": list(ML_MODELS.keys())}


@app.get("/models_list")
async def root():
    models_list, _ = get_models_list()
    return {"models_list": models_list}


@app.post("/create/{model_type}")
async def root(model_type: ModelType,
               model_params: RLParams | DTCParams | DTRParams | None = None
               ):
    if model_params is None:
        params = {}
    else:
        params = model_params.dict()

    ml_model = MLModel(type_model=model_type, params=params)
    model_path = ml_model.dump_model()
    return {"model_path": model_path}


@app.post("/fit/{model_name}")
async def root(model_name: str,
               uploaded_file: UploadFile = File(description="A file read as UploadFile")
               ):
    # contents = await uploaded_file.read()
    if uploaded_file.filename.split(".")[-1] == "xlsx" or uploaded_file.filename.split(".")[-1] == "xls":
        data = pd.read_csv(uploaded_file.file)
    elif uploaded_file.filename.split(".")[-1] == "csv":
        data = pd.read_csv(uploaded_file.file)
    # else:
    #     return TODO: return error code wrong format
    return {"column": list(data.columns)[0]}
