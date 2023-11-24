import numpy as np
import fastcat as fc

from my_utils import ds_corr

########################################
# Inputs
########################################
# Spectrum
E=100
Z=74 
# Inhirent filtration
z1=13 
x1=0.4
# External filtration
#z2=29
#x2=0.5
# Detector
detector="CsI-784-micrometer"
# Focal spot
fs=1.2
# Projections
th_f=np.pi*2
N=800
# Dose
dpp=0.04
# Bowtie filter
bowtief='bowtie_asym'
btf=False
# ASG
asg=False
# phantom
phantom_wos=fc.pelvis()
phantom_ws=fc.pelvis()
phantom_cnn=fc.pelvis()

muT_max=7
mfolder="./trained_models/fastcat_v2/unet_tl_pelvis01_"
outs='pelvis01'
########################################

angles = np.linspace(0,th_f,N,endpoint=False)

s = fc.calculate_spectrum(E,14,3,50,0,0.2,monitor=None,z=Z)
s.attenuate(x1,fc.get_mu(z1))
#s.attenuate(x2,fc.get_mu(z2))

det = fc.Detector(s,detector)
det.add_focal_spot(fs)

#phantom_wos.return_projs(det,s,angles,mgy=dpp,scat_on=False,ASG=asg,bowtie=btf,filter=bowtief)
#ofname='./proj_test/'+outs+'_woS_prj.npy'
#np.save(ofname,phantom_wos.intensity)
#phantom_wos.reconstruct('FDK',filt='ram_lak')
#ofname='./recon_test/'+outs+'_woS_im.npy'
#np.save(ofname,phantom_wos.img)

#phantom_ws.return_projs(det,s,angles,mgy=dpp,scat_on=True,ASG=asg,bowtie=btf,filter=bowtief)
#ofname='./proj_test/'+outs+'_wS_prj.npy'
#np.save(ofname,phantom_ws.intensity)
#phantom_ws.reconstruct('FDK',filt='ram_lak')
#ofname='./recon_test/'+outs+'_wS_im.npy'
#np.save(ofname,phantom_ws.img)

phantom_cnn.return_projs(det,s,angles,mgy=dpp,scat_on=True,ASG=asg,bowtie=btf,filter=bowtief)
p_corr,SR=ds_corr.proj_corr(mfolder+"model.json",mfolder+"weights.h5",phantom_cnn.intensity,phantom_cnn.flood,muT_max)
phantom_cnn.proj=p_corr
ofname='./proj_test/'+outs+'_cnn_prj.npy'
np.save(ofname,phantom_cnn.proj)
ofname='./proj_test/'+outs+'_cnn_SR.npy'
np.save(ofname,SR)
