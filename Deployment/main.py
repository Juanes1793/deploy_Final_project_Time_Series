from deployment_config import MODELS_DIR
import uvicorn
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from schemas import TimeSeriesInput
from utils import split_sequence, scaler
from functools import reduce

app = FastAPI()

time_series_model = tf.keras.models.load_model("time_series_model.h5")


# Endpoint for predictions
@app.post("/predict/")
async def predict(input_data: TimeSeriesInput):
    # Validate input length

    try: 
        if len(input_data.demanda_lista) < 96:  # `n_steps_in` = 96
            raise HTTPException(
                status_code=400,
                detail="The input list must contain at least 96 values."
            )
        
        # Preprocess input

        X, y = split_sequence(input_data.demanda_lista, 96, 1) 
        X_scaled = scaler.fit_transform(X)
        y = y.reshape(-1,1)
        y_scaled = scaler.fit_transform(y)
        n_features = 1 
        X_scaled = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], n_features))
        y_scaled = y_scaled.reshape((y_scaled.shape[0], y_scaled.shape[1], n_features)) 

        # Make predictionsc
        yhat = time_series_model.predict(X_scaled, verbose=0)
        Lista_predict = yhat.tolist()
        prediccion = reduce(lambda x,y: x+y, Lista_predict) 
        yreal_prediccion = scaler.inverse_transform(prediccion)
        yreal_prediccion = yreal_prediccion.tolist()
        yreal_prediccion1 = reduce(lambda x,y: x+y, yreal_prediccion)

        return {"prediccion": yreal_prediccion1}

    except Exception as e:
        return {"error": str(e)}




if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)


