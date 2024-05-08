import dopterian.dopterian as dopt
from astropy.io import fits
import numpy as np
import math

#Script para probar la ejecucion de dopterian

sci_image = r'D:\Documentos\Diego\U\Memoria Titulo\Dopterian\Input\clash_a209_nir_0990_dopterian_input.fits'
psf = r'D:\Documentos\Diego\U\Memoria Titulo\Dopterian\Input\hlsp_clash_hst_wfc3ir-65mas_all_f160w_v1_psf.fits'
sky_image = r'D:\Documentos\Diego\U\Memoria Titulo\Dopterian\Input\sky_clash_a209_nir_0990_a209_dopterian_input.fits'

output_sci = r'D:\Documentos\Diego\U\Memoria Titulo\Dopterian\Input\clash_a209_nir_0990_dopterian_outpu.fits'
output_psf = r'D:\Documentos\Diego\U\Memoria Titulo\Dopterian\Input\hlsp_clash_hst_wfc3ir-65mas_all_f160w_v1_output_psf.fits'


science_hdul = fits.open(sci_image)
science_data = science_hdul[0].data
science_header = science_hdul[0].header
science_hdul.close()

#exptime
exptime = science_header['EXPTIME']

#pixscale (need to check if this is correct)
CD1_1 = science_header['CD1_1']
CD1_2 = science_header['CD1_2']
CD2_1 = science_header['CD2_1']
CD2_2 = science_header['CD2_2']

pixscale = math.sqrt(CD1_1**2 + CD2_2**2)

#input_photflam and input_photplam

input_photflam = science_header['PHOTFLAM']
input_photplam = science_header['PHOTPLAM']
log_photflam = np.log10(input_photflam)
log_photplam = np.log10(input_photplam)

#zero point    
zero_point = -2.5 * log_photflam - 5.0 * log_photplam - 2.408


lowz_info  = {'redshift': 0.206, 'psf': psf,'zp': zero_point, 'exptime': exptime, 'filter': 'wfc3_f160w', 'lam_eff': input_photplam, 'pixscale': pixscale}

highz_info  = {'redshift': 2.0, 'psf': psf,'zp': zero_point, 'exptime': exptime, 'filter': 'wfc3_f160w', 'lam_eff': input_photplam, 'pixscale': pixscale}

imOUT, psfOUT = dopt.ferengi(sci_image,sky_image, lowz_info, highz_info, [output_sci, output_psf], imerr=None, noconv=False, evo=None)


#    imOUT,psfOUT = ferengi(InputImName,BgName,lowz_info,highz_info,['smooth_galpy_evo.fits','smooth_psfpy_evo.fits'],noconv=False,evo=lum_evolution)

#    fig,ax=mpl.subplots(1,2)
#    ax[0].imshow(imOUT)
#    ax[1].imshow(psfOUT)
#    mpl.show()
