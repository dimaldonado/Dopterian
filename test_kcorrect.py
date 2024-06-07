import dopterian.dopterian as dopt
from astropy.io import fits
import numpy as np
import math
import matplotlib.pyplot as plt
import kcorrect

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
science_hdul.info()
science_data = science_hdul[0].data
science_header = science_hdul[0].header
science_hdul.close()

n_pix_x = science_header['NAXIS1']
n_pix_y = science_header['NAXIS2']

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
plt.imshow(science_data, cmap='gray')
plt.title('Science Data Input')
plt.colorbar()


# Convertir la matriz de datos de science_data a float32
science_data = science_data.astype(np.float32)

# Convertir electrons/s a flujo físico en erg/s/cm^2/Å
flux_erg_per_s_cm2_A = science_data * input_photflam

# Convertir a flujo físico en erg/s/cm^2/Hz
c = 2.99792458e18  # Velocidad de la luz en Å/s
flux_erg_per_s_cm2_Hz = flux_erg_per_s_cm2_A * (input_photplam**2 / c)

# Convertir a maggies
flux_maggies = flux_erg_per_s_cm2_Hz / (3631e-23)

# Definir el error relativo en magnitudes (ejemplo con 0.2 magnitudes)
err = 0.2

# Calcular el error en maggies
err_maggies = flux_maggies * (np.log(10)/2.5) * err

# Asegurarse de que los errores sean finitos y reemplazar valores NaN/Inf
err_maggies = np.where(np.isfinite(err_maggies), err_maggies, np.nan)
err_maggies = np.nan_to_num(err_maggies, nan=1e10, posinf=1e10, neginf=1e10)

# Calcular la varianza inversa (ivar) y asegurarse de que sea finita
with np.errstate(divide='ignore', invalid='ignore'):
    ivar_maggies = 1 / err_maggies**2
    ivar_maggies = np.where(np.isfinite(ivar_maggies), ivar_maggies, np.nan)
    ivar_maggies = np.nan_to_num(ivar_maggies, nan=0.0, posinf=0.0, neginf=0.0)

# Convertir ivar_maggies y flux_maggies a float32
ivar_maggies = ivar_maggies.astype(np.float32)
flux_maggies = flux_maggies.astype(np.float32)

# Aplanar las matrices a 1D
ivar_maggies = ivar_maggies.flatten()
flux_maggies = flux_maggies.flatten()

# Número de píxeles
n_pizeles = len(flux_maggies)

# Definir un redshift constante para todos los píxeles (ejemplo con 0.206)
redshift = [0.206] * n_pizeles

# Definir las respuestas de los filtros utilizados
responses = ['clash_wfc3_f160w']

# Inicializar listas para almacenar los flujos y varianzas inversas por píxel
maggies = [[] for n in range(n_pizeles)]
ivar = [[] for n in range(n_pizeles)]

# Llenar las listas con los datos correspondientes
for i in range(n_pizeles):
    maggies[i].append(flux_maggies[i])
    ivar[i].append(ivar_maggies[i])

# Inicializar el objeto Kcorrect con las respuestas de los filtros
kc = kcorrect.kcorrect.Kcorrect(responses=responses)

# Calcular los coeficientes de ajuste para cada píxel
coeffs = kc.fit_coeffs(redshift=redshift, maggies=maggies, ivar=ivar)

# Calcular las correcciones k para cada píxel
k = kc.kcorrect(redshift=redshift, coeffs=coeffs)

# Calcular las magnitudes absolutas para cada píxel
absmag = kc.absmag(redshift=redshift, maggies=maggies, ivar=ivar, coeffs=coeffs)

absmag = np.clip(absmag, -np.finfo(np.float64).max, np.finfo(np.float64).max)

flux_transformed = 10**(-0.4 * (absmag + 48.6))

flux_transformed_image = flux_transformed.reshape((n_pix_y, n_pix_x))

plt.figure()
plt.imshow(flux_transformed_image, cmap='gray')
plt.title('Science Data Input')
plt.colorbar()
plt.show()