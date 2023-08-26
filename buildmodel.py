# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.model_selection import train_test_split
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras.layers import Dense,Dropout
#
# df=pd.read_csv(r'dataset.cvs')
# cols=list(df.columns)
#
# x=df.drop(["diameter"],axis=1)
# y=np.array(df["diameter"]).reshape((-1,1))
# scaler1=MinMaxScaler()
# scaler2=MinMaxScaler()
# x=scaler1.fit_transform(x)
# y=scaler2.fit_transform(y)
#
# xtrain,xtest,ytrain,ytest=train_test_split(x,y,random_state=42,train_size=0.8)
#
# from pickle import dump
#
# dump(scaler1, open('scaler1.pkl', 'wb'))
# dump(scaler2, open('scaler2.pkl', 'wb'))
#
# model=keras.Sequential([
#     Dense(128,activation="relu"),
#     Dropout(0.3),
#     Dense(64,activation="relu"),
#     Dropout(0.2),
#     Dense(32,activation="relu"),
#     Dense(1,activation="linear")
# ])
# model.compile(optimizer=keras.optimizers.Adam(),loss=keras.losses.mean_squared_error,metrics=[keras.metrics.RootMeanSquaredError()])
#
# model.fit(xtrain,ytrain,epochs=20,validation_split=0.1)
#
# print(model.evaluate(xtest,ytest))
#
# model.save("model1.h5")
#
