import kcorrect
import kcorrections.kcorrections as kk
import numpy as np
from astropy.io import fits
import dopterian.dopterian as dopt
import itertools
import dopterian.cosmology as cosmos
from astropy.cosmology import FlatLambdaCDM
import kcorrect as k
import matplotlib.pyplot as plt

#Script para probar la ejecucion de dopterian


catalog_path = [
                r'D:\Documentos\Diego\U\Memoria Titulo\Dopterian\Input\A209\A209_F160W_input_for_Dopterian.txt',
                r'D:\Documentos\Diego\U\Memoria Titulo\Dopterian\Input\A209\A209_F475W_input_for_Dopterian.txt',
                r'D:\Documentos\Diego\U\Memoria Titulo\Dopterian\Input\A209\A209_F625W_input_for_Dopterian.txt',
                r'D:\Documentos\Diego\U\Memoria Titulo\Dopterian\Input\A209\A209_F775W_input_for_Dopterian.txt',
                r'D:\Documentos\Diego\U\Memoria Titulo\Dopterian\Input\A209\A209_F814W_input_for_Dopterian.txt'
            ]

img = []
sky = []
rms = []
psf_lo = []
psf_hi = []
exptime_lo = []
input_photflam = []
input_photplam = []
log_photflam = []
log_photplam = []
zp_lo = []

#           -----------Low z parameters-----------


sci_path = [
    r'D:\Documentos\Diego\U\Memoria Titulo\Dopterian\Input\A209\SCI_F160W_clash_a209_nir_1767.fits',
    r'D:\Documentos\Diego\U\Memoria Titulo\Dopterian\Input\A209\SCI_F475W_clash_a209_nir_1767.fits',
    r'D:\Documentos\Diego\U\Memoria Titulo\Dopterian\Input\A209\SCI_F625W_clash_a209_nir_1767.fits',
    r'D:\Documentos\Diego\U\Memoria Titulo\Dopterian\Input\A209\SCI_F775W_clash_a209_nir_1767.fits',
    r'D:\Documentos\Diego\U\Memoria Titulo\Dopterian\Input\A209\SCI_F814W_clash_a209_nir_1767.fits',

]

sky_path = [
    r'D:\Documentos\Diego\U\Memoria Titulo\Dopterian\Input\A209\sky_F160W_clash_a209_nir_1767.fits',
    r'D:\Documentos\Diego\U\Memoria Titulo\Dopterian\Input\A209\sky_F475W_clash_a209_nir_1767.fits',
    r'D:\Documentos\Diego\U\Memoria Titulo\Dopterian\Input\A209\sky_F625W_clash_a209_nir_1767.fits',
    r'D:\Documentos\Diego\U\Memoria Titulo\Dopterian\Input\A209\sky_F775W_clash_a209_nir_1767.fits',
    r'D:\Documentos\Diego\U\Memoria Titulo\Dopterian\Input\A209\sky_F814W_clash_a209_nir_1767.fits',
]

rms_path = [
    r'D:\Documentos\Diego\U\Memoria Titulo\Dopterian\Input\A209\RMS_F160W_clash_a209_nir_1767.fits',
    r'D:\Documentos\Diego\U\Memoria Titulo\Dopterian\Input\A209\RMS_F475W_clash_a209_nir_1767.fits',
    r'D:\Documentos\Diego\U\Memoria Titulo\Dopterian\Input\A209\RMS_F625W_clash_a209_nir_1767.fits',
    r'D:\Documentos\Diego\U\Memoria Titulo\Dopterian\Input\A209\RMS_F775W_clash_a209_nir_1767.fits',
    r'D:\Documentos\Diego\U\Memoria Titulo\Dopterian\Input\A209\RMS_F814W_clash_a209_nir_1767.fits',

]

psf_path_list_lo = [ 
            r'D:\Documentos\Diego\U\Memoria Titulo\Dopterian\Input\psf\hlsp_clash_hst_wfc3ir-65mas_all_f160w_v1_psf.fits',
            r'D:\Documentos\Diego\U\Memoria Titulo\Dopterian\Input\psf\hlsp_clash_hst_acs-65mas_all_f475w_v1_psf.fits',
            r'D:\Documentos\Diego\U\Memoria Titulo\Dopterian\Input\psf\hlsp_clash_hst_acs-65mas_all_f625w_v1_psf.fits',
            r'D:\Documentos\Diego\U\Memoria Titulo\Dopterian\Input\psf\hlsp_clash_hst_acs-65mas_all_f775w_v1_psf.fits',
            r'D:\Documentos\Diego\U\Memoria Titulo\Dopterian\Input\psf\hlsp_clash_hst_acs-65mas_all_f814w_v1_psf.fits'
]

#low z filters

filter_lo = ['clash_wfc3_f160w','clash_wfc3_f475w','clash_wfc3_f625w','clash_wfc3_f775w','clash_wfc3_f814w']

#low z effective wavelengths
lambda_lo = [15405,4770,6310,7647,8057]

err0_mag = [0.05, 0.05, 0.05, 0.05, 0.05]

#reading images
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
    exptime_lo.append(science_hdul[0].header['EXPTIME'])
    input_photflam.append(science_hdul[0].header['PHOTFLAM'])
    input_photplam.append(science_hdul[0].header['PHOTPLAM'])
    log_photflam.append(np.log10(input_photflam[i]))
    log_photplam.append(np.log10(input_photplam[i]))
    zp_lo.append(-2.5 * log_photflam[i] - 5.0 * log_photplam[i] - 2.408)
    science_hdul.close()

pixscale_lo = 0.065




#           -----------High z parameters----------- 
psf_path_list_hi = [r'D:\Documentos\Diego\U\Memoria Titulo\Dopterian\Input\psf\hlsp_clash_hst_acs-65mas_all_f814w_v1_psf.fits']

filter_hi = ['clash_wfc3_f814w']

#high z effective wavelengths
lambda_hi= [8057]

#reading images
science_hdul = fits.open(psf_path_list_hi[0])
psf_lo.append(science_hdul[0].data)
science_hdul.close()

#high z zero point
zp_hi = zp_lo[-1]

#high z exposure time
exptime_hi = exptime_lo[-1]

#high z pixel scale
pixscale_hi = 0.065



lowz_info  = {'redshift': 0.202, 'psf': psf_path_list_lo,'zp': zp_lo, 'exptime': exptime_lo, 'filter': filter_lo, 'lam_eff': input_photplam, 'pixscale': pixscale_lo,'lambda': lambda_lo}

highz_info  = {'redshift': 2.0, 'psf': psf_path_list_hi,'zp': zp_hi, 'exptime': exptime_hi, 'filter': filter_hi, 'lam_eff': input_photplam, 'pixscale': pixscale_hi,'lambda': lambda_hi}


#output paths
output_sci = r'D:\Documentos\Diego\U\Memoria Titulo\Dopterian\Input\input kcorrect test\Output\SCI_F160W_clash_a209_nir_0798_output.fits'
output_psf = r'D:\Documentos\Diego\U\Memoria Titulo\Dopterian\Input\input kcorrect test\Output\SCI_F160W_clash_a209_nir_0798_output_psf.fits'

#os = FlatLambdaCDM(H0=cosmos.H0,Om0=cosmos.Omat,Ob0=cosmos.Obar)
#kc = k.kcorrect.Kcorrect(responses=filter_in, responses_out=filtter_out,responses_map=filter_map,cosmo=os)


imOUT,psfOUT= dopt.ferengi_k(sci_path, sky_path, lowz_info, highz_info, [output_sci, output_psf], imerr=rms_path, err0_mag=err0_mag, noconv=False, evo=None, nonoise=False, extend=False, noflux=True)

plt.figure()
plt.imshow(img[0],origin='lower', cmap='gray')
plt.title('Science Data Input')
plt.colorbar()
plt.show()


plt.figure()
plt.imshow(imOUT,origin='lower', cmap='gray')
plt.title('Science Data Input')
plt.colorbar()
plt.show()