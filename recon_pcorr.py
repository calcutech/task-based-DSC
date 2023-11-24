import numpy as np
import fastcat as fc
import tigre.algorithms as algs

phantom=fc.head()
outs='head01'

n_ang=800
ang_range=2*np.pi  
angles = np.linspace(0,ang_range, n_ang)

pcorr_fname='./proj_test/'+outs+'_cnn_prj.npy'
pcorr=np.load(pcorr_fname)

recon=algs.fdk(pcorr,phantom.geomet,angles)
ofname='./recon_test/'+outs+'_cnn_im.npy'
np.save(ofname,recon)
















