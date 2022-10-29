from fastapi import FastAPI, UploadFile, File
from ml_models import ML_MODELS, get_models_list, MLModel, convert_uploaded_file
# from enum_models import *
from api_models import *
# import pandas as pd

app = FastAPI()
# uvicorn main:app --reload


@app.get("/available_model_types")
async def root():
    return {"available_model_types": list(ML_MODELS.keys())}


@app.get("/models_list")
async def root():
    models_list, _ = get_models_list()
    return {"models_list": models_list}


@app.post("/models/{model_type}/create")
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


@app.post("/models/{model_name}/fit")
async def root(model_name: str,
               uploaded_file: UploadFile = File(description="A file read as UploadFile")
               ):

    data = convert_uploaded_file(uploaded_file)
    ml_model = MLModel(model_name=model_name)
    ml_model.fit(data)
    ml_model.dump_model()
    return {"is_fitted": ml_model.is_fitted}


@app.post("/models/{model_name}/predict")
async def root(model_name: str,
               uploaded_file: UploadFile = File(description="A file read as UploadFile")
               ):
    data = convert_uploaded_file(uploaded_file)
    ml_model = MLModel(model_name=model_name)
    result = ml_model.predict(data)
    return {"predict": list(result)}


@app.get("/models/{model_name}/info")
async def root(model_name: str):
    ml_model = MLModel(model_name=model_name)
    return ml_model.get_info()


@app.delete("/models/{model_name}/delete")
async def root(model_name: str):
    ml_model = MLModel(model_name=model_name)
    return ml_model.delete_model()
