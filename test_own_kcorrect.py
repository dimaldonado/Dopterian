import kcorrect
import kcorrections.kcorrections as kk
import numpy as np
from astropy.io import fits
import dopterian.dopterian as dopt
import itertools
import dopterian.cosmology as cosmos
from astropy.cosmology import FlatLambdaCDM
import kcorrect as k

#Script para probar la ejecucion de dopterian

sci_path = [
    r'D:\Documentos\Diego\U\Memoria Titulo\Dopterian\Input\input kcorrect test\SCI_F160W_clash_a209_nir_0798.fits',
    r'D:\Documentos\Diego\U\Memoria Titulo\Dopterian\Input\input kcorrect test\SCI_F475W_clash_a209_nir_0798.fits',
    r'D:\Documentos\Diego\U\Memoria Titulo\Dopterian\Input\input kcorrect test\SCI_F625W_clash_a209_nir_0798.fits',
    r'D:\Documentos\Diego\U\Memoria Titulo\Dopterian\Input\input kcorrect test\SCI_F775W_clash_a209_nir_0798.fits',
    r'D:\Documentos\Diego\U\Memoria Titulo\Dopterian\Input\input kcorrect test\SCI_F814W_clash_a209_nir_0798.fits',

]

sky_path = [
    r'D:\Documentos\Diego\U\Memoria Titulo\Dopterian\Input\input kcorrect test\sky_F160W_clash_a209_nir_0798.fits',
    r'D:\Documentos\Diego\U\Memoria Titulo\Dopterian\Input\input kcorrect test\sky_F475W_clash_a209_nir_0798.fits',
    r'D:\Documentos\Diego\U\Memoria Titulo\Dopterian\Input\input kcorrect test\sky_F625W_clash_a209_nir_0798.fits',
    r'D:\Documentos\Diego\U\Memoria Titulo\Dopterian\Input\input kcorrect test\sky_F775W_clash_a209_nir_0798.fits',
    r'D:\Documentos\Diego\U\Memoria Titulo\Dopterian\Input\input kcorrect test\sky_F814W_clash_a209_nir_0798.fits',
]

rms_path = [
    r'D:\Documentos\Diego\U\Memoria Titulo\Dopterian\Input\input kcorrect test\RMS_F160W_clash_a209_nir_0798.fits',
    r'D:\Documentos\Diego\U\Memoria Titulo\Dopterian\Input\input kcorrect test\RMS_F475W_clash_a209_nir_0798.fits',
    r'D:\Documentos\Diego\U\Memoria Titulo\Dopterian\Input\input kcorrect test\RMS_F625W_clash_a209_nir_0798.fits',
    r'D:\Documentos\Diego\U\Memoria Titulo\Dopterian\Input\input kcorrect test\RMS_F775W_clash_a209_nir_0798.fits',
    r'D:\Documentos\Diego\U\Memoria Titulo\Dopterian\Input\input kcorrect test\RMS_F814W_clash_a209_nir_0798.fits',

]

psf_path_list_lo = [ 
            r'D:\Documentos\Diego\U\Memoria Titulo\Dopterian\Input\psf\hlsp_clash_hst_wfc3ir-65mas_all_f160w_v1_psf.fits',
            r'D:\Documentos\Diego\U\Memoria Titulo\Dopterian\Input\psf\hlsp_clash_hst_acs-65mas_all_f475w_v1_psf.fits',
            r'D:\Documentos\Diego\U\Memoria Titulo\Dopterian\Input\psf\hlsp_clash_hst_acs-65mas_all_f625w_v1_psf.fits',
            r'D:\Documentos\Diego\U\Memoria Titulo\Dopterian\Input\psf\hlsp_clash_hst_acs-65mas_all_f775w_v1_psf.fits',
            r'D:\Documentos\Diego\U\Memoria Titulo\Dopterian\Input\psf\hlsp_clash_hst_acs-65mas_all_f814w_v1_psf.fits'
]

psf_path_list_hi = [r'D:\Documentos\Diego\U\Memoria Titulo\Dopterian\Input\psf\hlsp_clash_hst_acs-65mas_all_f814w_v1_psf.fits']

output_sci = r'D:\Documentos\Diego\U\Memoria Titulo\Dopterian\Input\input kcorrect test\SCI_F160W_clash_a209_nir_0798_output.fits'
output_psf = r'D:\Documentos\Diego\U\Memoria Titulo\Dopterian\Input\input kcorrect test\SCI_F160W_clash_a209_nir_0798_output_psf.fits'

filter_in = ['clash_wfc3_f160w','clash_wfc3_f475w','clash_wfc3_f625w','clash_wfc3_f775w','clash_wfc3_f814w']
lambda_lo = [15405,4770,6310,7647,8057]

filter_map = ['clash_wfc3_f814w']

err0_mag = [0.05, 0.05, 0.05, 0.05, 0.05]
filtter_out = ['clash_wfc3_f814w']
lambda_hi= [8057]

img = []
sky = []
rms = []
psf_lo = []
psf_hi = []
exptime = []
input_photflam = []
input_photplam = []
log_photflam = []
log_photplam = []
zero_point = []

for i in range(len(sci_path)):
    science_hdul = fits.open(sci_path[i])
    img.append(science_hdul[0].data)
    science_hdul.close()

    science_hdul = fits.open(sky_path[i])
    sky.append(science_hdul[0].data)
    science_hdul.close()

    science_hdul = fits.open(rms_path[i])
    rms.append(science_hdul[0].data)
    science_hdul.close()

    science_hdul = fits.open(psf_path_list_lo[i])
    psf_lo.append(science_hdul[0].data)
    science_hdul.close()

    science_hdul = fits.open(sci_path[i])
    exptime.append(science_hdul[0].header['EXPTIME'])
    input_photflam.append(science_hdul[0].header['PHOTFLAM'])
    input_photplam.append(science_hdul[0].header['PHOTPLAM'])
    log_photflam.append(np.log10(input_photflam[i]))
    log_photplam.append(np.log10(input_photplam[i]))
    zero_point.append(-2.5 * log_photflam[i] - 5.0 * log_photplam[i] - 2.408)
    science_hdul.close()

science_hdul = fits.open(psf_path_list_hi[0])
psf_lo.append(science_hdul[0].data)
science_hdul.close()
#ultimo elemento de zero_point
zp_hi = zero_point[-1]
exptime_hi = exptime[-1]

pixscale = 0.065

lowz_info  = {'redshift': 0.194, 'psf': psf_path_list_lo,'zp': zero_point, 'exptime': exptime, 'filter': filter_in, 'lam_eff': input_photplam, 'pixscale': pixscale,'lambda': lambda_lo}

highz_info  = {'redshift': 2.0, 'psf': psf_path_list_hi,'zp': zp_hi, 'exptime': exptime_hi, 'filter': filtter_out, 'lam_eff': input_photplam, 'pixscale': pixscale,'lambda': lambda_hi}

#os = FlatLambdaCDM(H0=cosmos.H0,Om0=cosmos.Omat,Ob0=cosmos.Obar)
#kc = k.kcorrect.Kcorrect(responses=filter_in, responses_out=filtter_out,responses_map=filter_map,cosmo=os)


imOUT= dopt.ferengi_k(sci_path, sky_path, lowz_info, highz_info, [output_sci, output_psf], imerr=rms_path, err0_mag=err0_mag, noconv=False, evo=None, nonoise=True, extend=False, noflux=True)