from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd
import tensorflow as tf

app=FastAPI()

class Asteroid(BaseModel):
        a: float
        e: float
        i: float
        om: float
        w: float
        q: float
        ad: float
        per_y: float
        data_arc: float
        condition_code: float
        n_obs_used: int
        H: float
        spkid: int
        sats: int
        albedo: float
        diameter_sigma: float
        orbit_id: float
        epoch: float
        epoch_mjd: int
        epoch_cal: int
        ma: float
        n: float
        tp: float
        per: float
        moid: float
        moid_ld: float
        moid_jup: float
        t_jup: float
        sigma_e: float
        sigma_a: float
        sigma_q: float
        sigma_i: float
        sigma_om: float
        sigma_w: float
        sigma_ma: float
        sigma_ad: float
        sigma_n: float
        sigma_tp: float
        sigma_per: float
        rms: float

with open('scaler1.pkl','rb') as f:
        scaler1=pickle.load(f)
with open('scaler2.pkl','rb') as f:
        scaler2=pickle.load(f)
model=tf.keras.models.load_model('model1.h5')

@app.post('/')
async def diameter_endpoint(item:Asteroid):
        # return item
        df=scaler1.transform(pd.DataFrame([item.dict().values()],columns=item.dict().keys()))
        pred=scaler2.inverse_transform(model.predict(df))
        return {"diameter":float(pred)}