import dopterian.dopterian as dopt
from astropy.io import fits
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
import os
import matplotlib.colors as mcolors

#  Leer el archivo de texto para obtener los datos del catálogo filtro F160W

catalog_path = [
                r'D:\Documentos\Diego\U\Memoria Titulo\Dopterian\Input\A209\A209_F160W_input_for_Dopterian.txt',
                r'D:\Documentos\Diego\U\Memoria Titulo\Dopterian\Input\A209\A209_F475W_input_for_Dopterian.txt',
                r'D:\Documentos\Diego\U\Memoria Titulo\Dopterian\Input\A209\A209_F625W_input_for_Dopterian.txt',
                r'D:\Documentos\Diego\U\Memoria Titulo\Dopterian\Input\A209\A209_F775W_input_for_Dopterian.txt',
                r'D:\Documentos\Diego\U\Memoria Titulo\Dopterian\Input\A209\A209_F814W_input_for_Dopterian.txt'
            ]

psf_path_list = { 
            'F160' : r'D:\Documentos\Diego\U\Memoria Titulo\Dopterian\Input\psf\hlsp_clash_hst_wfc3ir-65mas_all_f160w_v1_psf.fits',
            'F475' : r'D:\Documentos\Diego\U\Memoria Titulo\Dopterian\Input\psf\hlsp_clash_hst_wfc3ir-65mas_all_f160w_v1_psf.fits',
            'F625' : r'D:\Documentos\Diego\U\Memoria Titulo\Dopterian\Input\psf\hlsp_clash_hst_wfc3ir-65mas_all_f160w_v1_psf.fits',
            'F775' : r'D:\Documentos\Diego\U\Memoria Titulo\Dopterian\Input\psf\hlsp_clash_hst_wfc3ir-65mas_all_f160w_v1_psf.fits',
            'F814' : r'D:\Documentos\Diego\U\Memoria Titulo\Dopterian\Input\psf\hlsp_clash_hst_wfc3ir-65mas_all_f160w_v1_psf.fits'
        }
filters= [
            "F160",
            "F475",
            "F625",
            "F775",
            "F814"
        ]


input_image_list = []
output_image_list = []
data_list = []

# Procesar cada fila del catálogo
for i in range(len(catalog_path)):
    
    catalog = pd.read_csv(catalog_path[i], delim_whitespace=True)
    
    for index, row in catalog.iterrows():
        clashid = row['CLASHID']
        cluster_name = row['clusterName']
        zb_1 = row['zb_1']
        
        
        # Crear los nombres de los archivos FITS
        base_filename = f"{clashid}.fits"

        sky_filename = f"sky_{filters[i]}W_{base_filename}"
        rms_filename = f"RMS_{filters[i]}W_{base_filename}"
        sci_filename = f"SCI_{filters[i]}W_{base_filename}"

        print(sci_filename+"------------------------------------")
        print(sky_filename)
        print(rms_filename)

        
        sky_fits_path = os.path.join('D:/Documentos/Diego/U/Memoria Titulo/Dopterian/Input/A209/', sky_filename)
        rms_fits_path = os.path.join('D:/Documentos/Diego/U/Memoria Titulo/Dopterian/Input/A209/', rms_filename)
        sci_fits_path = os.path.join('D:/Documentos/Diego/U/Memoria Titulo/Dopterian/Input/A209/', sci_filename)
        psf_path = os.path.join('D:/Documentos/Diego/U/Memoria Titulo/Dopterian/Input/A209/', psf_path_list[filters[i]])

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
            print(f"Archivo no encontrado: {psf_path[filters[i]]}")
            continue
        
        
        if sky_ok and rms_ok and sci_ok and psf_ok:
            output_sci = os.path.join('D:/Documentos/Diego/U/Memoria Titulo/Dopterian/Input/A209/output', sci_filename)
            output_psf = os.path.join('D:/Documentos/Diego/U/Memoria Titulo/Dopterian/Input/A209/output', 'psf_'+sci_filename)

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

            filter = "wfc3_"+filters[i].lower()

            lowz_info  = {'redshift': zb_1, 'psf': psf_path,'zp': zero_point, 'exptime': exptime, 'filter': 'wfc3_f160w', 'lam_eff': input_photplam, 'pixscale': pixscale}

            highz_info  = {'redshift': 2.0, 'psf': psf_path,'zp': zero_point, 'exptime': exptime, 'filter': 'wfc3_f160w', 'lam_eff': input_photplam, 'pixscale': pixscale}

            input_image_list.append(science_data)
            data_list.append(sci_filename+"  z:"+str(zb_1))
            imOUT, psfOUT = dopt.ferengi(sci_fits_path, sky_fits_path, lowz_info, highz_info, [output_sci, output_psf], imerr=rms_fits_path ,noflux=False,evo=None, noconv=False, kcorrect=False, extend=False, nonoise=True)
            output_image_list.append(imOUT)

            fig, axs = plt.subplots(1, 2, figsize=(10, 5))
            
            im1 = axs[0].imshow(science_data, cmap='gray')
            axs[0].set_title('Input Image:'+"  z:"+str(zb_1)) 
            plt.colorbar(im1, ax=axs[0])

            im2 = axs[1].imshow(imOUT, cmap='gray')
            axs[1].set_title('Output Image'+" z: 2.0")
            plt.colorbar(im2, ax=axs[1])
            
            plt.suptitle(f'Comparación de Imágenes: {sci_filename}', fontsize=16)
            # Guardar la figura
            comparison_image_path = os.path.join('D:/Documentos/Diego/U/Memoria Titulo/Dopterian/Input/A209/output/', f'comparison_{sci_filename}.png')
            plt.savefig(comparison_image_path)
            plt.close(fig)
    