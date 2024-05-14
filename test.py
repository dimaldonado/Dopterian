import dopterian.dopterian as dopt
from astropy.io import fits
import numpy as np
import math
import matplotlib.pyplot as plt

#Script para probar la ejecucion de dopterian

sci_image = r'D:\Documentos\Diego\U\Memoria Titulo\Dopterian\Input\clash_a209_nir_0990_dopterian_input.fits'
psf = r'D:\Documentos\Diego\U\Memoria Titulo\Dopterian\Input\hlsp_clash_hst_wfc3ir-65mas_all_f160w_v1_psf.fits'
sky_image = r'D:\Documentos\Diego\U\Memoria Titulo\Dopterian\Input\sky_clash_a209_nir_0990_a209_dopterian_input.fits'

output_sci = r'D:\Documentos\Diego\U\Memoria Titulo\Dopterian\Output\clash_a209_nir_0990_dopterian_outpu.fits'
output_psf = r'D:\Documentos\Diego\U\Memoria Titulo\Dopterian\Output\hlsp_clash_hst_wfc3ir-65mas_all_f160w_v1_output_psf.fits'

#leemos los fits

science_hdul = fits.open(sky_image)
sky_data = science_hdul[0].data
science_hdul.close()

science_hdul = fits.open(psf)
psf_data = science_hdul[0].data
science_hdul.close()

science_hdul = fits.open(sci_image)
science_data = science_hdul[0].data
science_header = science_hdul[0].header
science_hdul.close()

#exptime
exptime = science_header['EXPTIME']

#pixscale 

pixscale =  0.065
 
#input_photflam and input_photplam

input_photflam = science_header['PHOTFLAM']
input_photplam = science_header['PHOTPLAM']
log_photflam = np.log10(input_photflam)
log_photplam = np.log10(input_photplam)

#zero point    
zero_point = -2.5 * log_photflam - 5.0 * log_photplam - 2.408


lowz_info  = {'redshift': 0.206, 'psf': psf,'zp': zero_point, 'exptime': exptime, 'filter': 'wfc3_f160w', 'lam_eff': input_photplam, 'pixscale': pixscale}

highz_info  = {'redshift': 2.0, 'psf': psf,'zp': zero_point, 'exptime': exptime, 'filter': 'wfc3_f160w', 'lam_eff': input_photplam, 'pixscale': pixscale}

#graficamos las imagenes de entrada

plt.figure()
plt.imshow(sky_data, cmap='gray')
plt.title('Sky Data Input')
plt.colorbar()


plt.figure()
plt.imshow(science_data, cmap='gray')
plt.title('Science Data Input')
plt.colorbar()

plt.figure()
plt.imshow(psf_data, cmap='gray')
plt.title('Psf Data Input')
plt.colorbar()


#ejecutamos dopterian
imOUT, psfOUT = dopt.ferengi(sci_image,sky_image, lowz_info, highz_info, [output_sci, output_psf], imerr=None, noconv=False, evo=None)

#graficamos las imagenes de salida
plt.figure()
plt.imshow(imOUT, cmap='gray')
plt.title('imOUT')
plt.colorbar()

plt.figure()
plt.imshow(psfOUT, cmap='gray')
plt.title('psfOUT')
plt.colorbar()
plt.show()
