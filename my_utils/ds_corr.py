from tensorflow.keras.models import Sequential, model_from_json
import numpy as np

def proj_corr(model_json,weights_h5,I,flood,muT_max):
    f=1.3
    p=log_proj(I,flood)   
    p=p/muT_max
    unet_model=load_model(model_json,weights_h5)
    SR=unet_model.predict(p)
    SR=SR.squeeze()*f
    I_corr = I * (1 - SR)
    p_corr=log_proj(I_corr,flood)
    return p_corr, SR 

def load_model(model_json,weights_h5):
    json_file = open(model_json, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(weights_h5)
    return loaded_model

def log_proj(I,flood):
    I0=flood[np.newaxis,np.newaxis,:]
    I0=np.repeat(I0, I.shape[0], axis=0)
    I0=np.repeat(I0, I.shape[1], axis=1)    
    p = np.log(I0 / I)
    p[p < 0] = 0.0001
    p[np.isnan(p)] = 0.0001
    p[np.isinf(p)] = 0.0001
    return p




