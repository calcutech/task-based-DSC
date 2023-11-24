from my_utils import impair_gen
from my_utils import unet_arch
from my_utils import ds_corr
from tensorflow import keras

import tensorflow as tf
import os
import numpy as np
import gc

data_folder='tl_pelvis08'
label="fastcat_v2/unet_tl_pelvis08_"
weights_dir="./trained_models/fastcat_v2/unet_pretrain_"

train_input_dir = './data/' + data_folder + '/train_input'
train_target_dir = './data/' + data_folder + '/train_target'
val_input_dir = './data/' + data_folder + '/val_input'
val_target_dir = './data/' + data_folder + '/val_target'

nx = 512
ny = 256
nz = 1

seed = 1789
bs = 20
n_ep = 50

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
unet_model=ds_corr.load_model(weights_dir+"model.json",weights_dir+"weights.h5")

for layer in unet_model.layers:
    if layer.name in ["conv2d_24", "conv2d_25", "conv2d_26"]:
        layer.trainable = False

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
