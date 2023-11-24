from my_utils import impair_gen
from my_utils import ds_corr
import numpy as np

mfolder="./trained_models/fastcat_v2/unet_pretrain_"
ofolder="./proj_test/pretrain_"
test_input_dir = "./data/pre_train/test_input/"
test_target_dir = "./data/pre_train/test_target/"
nx = 512
ny = 256
bs = 777

unet_model=ds_corr.load_model(mfolder+"model.json",mfolder+"weights.h5")

test_pair_gen = impair_gen.impairGenerator(test_input_dir,test_target_dir,nx,ny,bs,0)
x,y=test_pair_gen.__next__()
yp=unet_model.predict(x)

np.save(ofolder+"test_input",x)
np.save(ofolder+"test_target",y)
np.save(ofolder+"test_pred",yp)
