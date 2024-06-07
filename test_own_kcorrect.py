import kcorrect
import kcorrections.kcorrections as kk
import numpy as np
from astropy.io import fits


sci_image = r'D:\Documentos\Diego\U\Memoria Titulo\Dopterian\Input\clash_a209_nir_0990_dopterian_input.fits'
science_hdul = fits.open(sci_image)
science_data = science_hdul[0].data
science_header = science_hdul[0].header
science_hdul.close()


redshift = 0.2
imerr= [1/np.sqrt(np.abs(science_data))]*4
data_filter_list = [science_data,science_data,science_data,science_data]
filter_list = ['sdss_u0', 'sdss_g0', 'sdss_r0','sdss_i0']

kk.kcorrect_image(data_filter_list,filter_list,redshift,imerr)