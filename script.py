from astropy.io import fits

# Ruta al archivo FITS original
input_file_path = r'D:\Documentos\Diego\U\Memoria Titulo\Dopterian\Input\psf\A209_Rc_PSFex_psf.fits'

# Ruta para guardar el nuevo archivo FITS
output_file_path = r'D:\Documentos\Diego\U\Memoria Titulo\Dopterian\Input\psf\edit_A209_Rc_PSFex_psf.fits'
# Leer el archivo FITS original
hdul = fits.open(input_file_path)
data = hdul[0].data
header = hdul[0].header

# Remover la dimensión extra
if data.shape[0] == 1:
    data = data[0]

# Crear un nuevo HDU con los datos modificados y el mismo encabezado
new_hdu = fits.PrimaryHDU(data, header=header)

# Escribir el nuevo archivo FITS
new_hdul = fits.HDUList([new_hdu])
new_hdul.writeto(output_file_path, overwrite=True)

# Cerrar los archivos FITS
hdul.close()
new_hdul.close()

print(f"Archivo FITS modificado guardado en {output_file_path}")