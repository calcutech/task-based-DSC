from my_utils import impair_gen
from my_utils import unet_arch
from tensorflow import keras

import tensorflow as tf
import os
import numpy as np
import gc

data_folder='mix'
label="fastcat_v2/unet_mix_"

train_input_dir = './data/' + data_folder + '/train_input'
train_target_dir = './data/' + data_folder + '/train_target'
val_input_dir = './data/' + data_folder + '/val_input'
val_target_dir = './data/' + data_folder + '/val_target'

nx = 512
ny = 256
nz = 1

seed = 1654
bs = 30
n_ep = 150

LR=1e-4

def SSIMLoss(y_true, y_pred):
    return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))
loss_func=SSIMLoss
optim=tf.keras.optimizers.Adam(learning_rate=LR)
perf_metric=['mse']

N_train = len(os.listdir(train_input_dir + "/train"))
N_val = len(os.listdir(val_input_dir + "/val"))
spe_train = N_train//bs
spe_val = N_val//bs

train_pair_gen = impair_gen.impairGenerator(train_input_dir,train_target_dir,nx,ny,bs,seed)
val_pair_gen = impair_gen.impairGenerator(val_input_dir,val_target_dir,nx,ny,bs,seed)

keras.backend.clear_session()
unet_model = unet_arch.build_unet_model(ny,nx,nz)
unet_model.summary()
unet_model.compile(optimizer=optim,loss=loss_func,metrics=perf_metric)
history=unet_model.fit(train_pair_gen,
          steps_per_epoch=spe_train,
          epochs=n_ep,
          verbose=1,
          validation_data=val_pair_gen,
          validation_steps=spe_val)

loss = history.history['loss']
val_loss = history.history['val_loss'] 

aux = open("./trained_models/"+label+"loss.txt", "w")
for element in loss:
    aux.write(str(element) + "\n")
aux.close()

aux = open("./trained_models/"+label+"vloss.txt", "w")
for element in val_loss:
    aux.write(str(element) + "\n")
aux.close()

unet_model.save_weights("./trained_models/"+label+"weights.h5")
with open("./trained_models/"+label+"model.json",'w') as f:
    f.write(unet_model.to_json())

gc.collect()
