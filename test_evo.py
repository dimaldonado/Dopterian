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

print(science_data.ndim)

#graficamos las imagenes de entrada
''''
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
'''

# Lista para almacenar las imágenes de salida
imOUT_list = []

# Ejecutamos dopterian con evo=None y almacenamos el resultado
imOUT, psfOUT = dopt.ferengi(sci_image, sky_image, lowz_info, highz_info, [output_sci, output_psf], imerr=None, noconv=False, evo=None, nonoise=True, extend=False, noflux=True)
imOUT_list.append(('evo=lum_factor()', imOUT))

# Ejecutamos dopterian con evo desde -1 hasta 1 en saltos de 0.1
for evo in np.arange(-1, 1.1, 0.1):
    imOUT, psfOUT = dopt.ferengi(sci_image, sky_image, lowz_info, highz_info, [output_sci, output_psf], imerr=None, noconv=False, evo=evo, nonoise=True, extend=False, noflux=True)
    imOUT_list.append((f'evo={evo:.1f}', imOUT))

# Encontramos los valores mínimos y máximos para todas las imágenes
vmin = min(im[1].min() for im in imOUT_list)
vmax = max(im[1].max() for im in imOUT_list)

# Crear una figura y ejes para todas las imágenes
ncols = 5  # Número de columnas para la cuadrícula
nrows = int(np.ceil(len(imOUT_list) / ncols))  # Calculamos el número de filas
fig, axs = plt.subplots(nrows, ncols, figsize=(20, 4 * nrows))

# Graficamos las imágenes de salida en los subplots
for ax, (title, imOUT) in zip(axs.flat, imOUT_list):
    im = ax.imshow(imOUT, cmap='gray', vmin=vmin, vmax=vmax)
     
    ax.set_title(title)
    
    fig.colorbar(im, ax=ax)

# Eliminar los ejes vacíos si el número de subplots es mayor que el número de imágenes
for ax in axs.flat[len(imOUT_list):]:
    fig.delaxes(ax)

plt.tight_layout()
plt.show()
