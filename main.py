from fastapi import Security, FastAPI, Depends, HTTPException
from fastapi.security.api_key import APIKey, APIKeyQuery
from fastapi.middleware.cors import CORSMiddleware
from starlette.status import HTTP_403_FORBIDDEN
from pydantic import BaseModel
import uvicorn
import pickle

API_KEY = "1234567asdfgh"
API_KEY_NAME = "access_token"

api_key_query = APIKeyQuery(name=API_KEY_NAME, auto_error=False)

app = FastAPI(title="API para generar predicciones del modelo Iris",
              description="Esta api tiene un endpoint para obtener que tipo de flor"
                          " Iris se obtiene con base en las características de la misma. Siuuuuuuuuuuuu",
              version="0.0.1")


origins = ["*"]  # Mala práctica, hay que limitar las IP.

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_api_key(api_key_query: str = Security(api_key_query)):
    if api_key_query == API_KEY:
        return api_key_query
    else:
        raise HTTPException(
            status_code=HTTP_403_FORBIDDEN,
            detail="Could not validate credentials")


class Iris(BaseModel):
    pl: float
    pw: float
    sl: float
    sw: float


# Indicación para que se ejecute en cuanto inicie la api
@app.on_event("startup")
def load_model():
    global model_iris  # Función para que sea global la variable
    with open(r"./model/modelIris.pickle", "rb") as f:
        model_iris = pickle.load(f)


@app.get("/")
def home():
    return{"Desc": "Health Check"}


@app.get("/api/v1/classify")
def classify_iris(iris: Iris):#, APIKey=Depends(get_api_key)):
    params = [[iris.sl, iris.sw, iris.pl, iris.pw]]
    pred = model_iris.predict(params)
    dict_iris = {0: "Setosa",
                 1: "Versicolor",
                 2: "Virginica"}
    return {"Iris flower": dict_iris.get(pred[0]),
            "Desc": "Predicción hecha correctamente"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="localhost", port=8000, reload=False)
