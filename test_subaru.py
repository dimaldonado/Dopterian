import pandas as pd
import os
from astropy.io import fits
from dopterian import dopterian as dopt

catalog_path = r'D:\Documentos\Diego\U\Memoria Titulo\Dopterian\Input\A209\A209_F160W_input_for_Dopterian.txt'

psf_path = r'D:\Documentos\Diego\U\Memoria Titulo\Dopterian\Input\psf\edit_A209_Rc_PSFex_psf.fits'

filter = 'subaru_suprimecam_Rc'

catalog = pd.read_csv(catalog_path, delim_whitespace=True)

ok = 0 
    
for index, row in catalog.iterrows():
    clashid = row['CLASHID']
    cluster_name = row['clusterName']
    zb_1 = row['zb_1']
    
    
    # Crear los nombres de los archivos FITS
    base_filename = f"{clashid}.fits"

    sky_filename = f"sky_Rc_{base_filename}"
    rms_filename = f"RMS_Rc_{base_filename}"
    sci_filename = f"SCI_Rc_{base_filename}"

    print(sci_filename+"------------------------------------")
    print(sky_filename)
    print(rms_filename)

    
    sky_fits_path = os.path.join('D:/Documentos/Diego/U/Memoria Titulo/Dopterian/Input/A209/', sky_filename)
    rms_fits_path = os.path.join('D:/Documentos/Diego/U/Memoria Titulo/Dopterian/Input/A209/', rms_filename)
    sci_fits_path = os.path.join('D:/Documentos/Diego/U/Memoria Titulo/Dopterian/Input/A209/', sci_filename)
    print(psf_path)

    # Leer los datos de los archivos FITS
    sky_ok = False
    rms_ok = False
    sci_ok = False

    try:
        with fits.open(sky_fits_path) as hdul:
            sky_data = hdul[0].data
            sky_ok = True
    except FileNotFoundError:
        print(f"Archivo no encontrado: {sky_fits_path}")
        continue

    try:
        with fits.open(rms_fits_path) as hdul:
            imerr_data = hdul[0].data
            rms_ok = True
    except FileNotFoundError:
        print(f"Archivo no encontrado: {rms_fits_path}")
        continue

    try:
        with fits.open(sci_fits_path) as hdul:
            science_data = hdul[0].data
            science_header = hdul[0].header
            sci_ok = True
    except FileNotFoundError:
        print(f"Archivo no encontrado: {sci_fits_path}")
        continue

    try:
        with fits.open(psf_path) as hdul:
            psf_data = hdul[0].data
            psf_ok = True
    except FileNotFoundError:
        print(f"Archivo no encontrado: {psf_path}")
        continue

    if sky_ok and rms_ok and sci_ok and psf_ok:
        exptime = 2400.0 
        zp = 27.46
        pixscale = 0.2
        lambda_lo = 6550
        err0_mag = [0.05]

        lowz_info  = {'redshift': zb_1,
                  'psf': [psf_path],
                  'zp': [zp],
                  'exptime': [exptime],
                  'filter': [filter], 
                  'pixscale': pixscale,
                  'lambda': [lambda_lo]}

        highz_info  = {'redshift': 2.0, 
                    'psf': psf_path,
                    'zp': zp, 
                    'exptime': exptime,
                    'filter': filter, 
                    'pixscale': pixscale,
                    'lambda': lambda_lo}
    
        name = sci_filename[4:]
        output_sci = "D:\\Documentos\\Diego\\U\\Memoria Titulo\Dopterian\\Input\\A209\\test_rc_to_rc_z2\\"+"output_sci_"+name
        output_psf = "D:\\Documentos\\Diego\\U\\Memoria Titulo\Dopterian\\Input\\A209\\test_rc_to_rc_z2\\"+"output_psf_"+name
        
        imOUT,psfOUT,n_pkcorrect= dopt.ferengi(
                                images = [sci_fits_path],
                                background= [sky_fits_path],
                                lowz_info = lowz_info, 
                                highz_info = highz_info, 
                                namesout= [output_sci, output_psf], 
                                imerr = [rms_fits_path],
                                err0_mag = err0_mag, 
                                noconv=False, 
                                evo=None, 
                                nonoise=True, 
                                extend=False, 
                                noflux=True)



