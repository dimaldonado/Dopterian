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
import pandas as pd
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable 

#Script para probar la ejecucion de dopterian + kcorrect

catalogs = {
            "F160W":    r'D:\Documentos\Diego\U\Memoria Titulo\Dopterian\Input\A209\A209_F160W_input_for_Dopterian.txt',
            "F475W":   r'D:\Documentos\Diego\U\Memoria Titulo\Dopterian\Input\A209\A209_F475W_input_for_Dopterian.txt',
            "F625W":    r'D:\Documentos\Diego\U\Memoria Titulo\Dopterian\Input\A209\A209_F625W_input_for_Dopterian.txt',
            "F775W":    r'D:\Documentos\Diego\U\Memoria Titulo\Dopterian\Input\A209\A209_F775W_input_for_Dopterian.txt',
            "F814W":   r'D:\Documentos\Diego\U\Memoria Titulo\Dopterian\Input\A209\A209_F814W_input_for_Dopterian.txt'
            
            }
data = {}

# Cargar los datos relevantes
for key, file in catalogs.items():
    data[key] = pd.read_csv(file, delim_whitespace=True, usecols=["CLASHID", "zb_1", "clusterName"])

# Obtener los nombres de los archivos que están presentes en todos los filtros
common_clashids = set(data["F160W"]["CLASHID"])

for key in data:
    common_clashids.intersection_update(data[key]["CLASHID"])

# Crear las listas de listas
input_image_lists = [[] for _ in range(5)]
z = [[] for _ in range(5)]
clustername = [[] for _ in range(5)]

# Asignar los nombres de los archivos correspondientes a cada filtro, así como zb_1 y clusterName
filters = ["F160W", "F475W", "F625W", "F775W", "F814W"]

for idx, filter in enumerate(filters):
    for clash_id in common_clashids:
        row = data[filter][data[filter]["CLASHID"] == clash_id]
        if not row.empty:
            input_image_lists[idx].append(f"{filter}_{row['CLASHID'].values[0]}.fits")
            z[idx].append(row["zb_1"].values[0])
            clustername[idx].append(row["clusterName"].values[0])

base_path = "D:\\Documentos\\Diego\\U\\Memoria Titulo\\Dopterian\\Input\\A209\\"


filename = input_image_lists
sci_images_path = [[base_path+"SCI_" + image for image in sublist] for sublist in input_image_lists]
rms_images_path = [[base_path+"RMS_" + image for image in sublist] for sublist in input_image_lists]
sky_images_path = [[base_path+"sky_" + image for image in sublist] for sublist in input_image_lists]

science_data = [[] for _ in range(5)]
id_g= [[] for _ in range(5)]
input_image_path = [[] for _ in range(5)]
input_rms_path = [[] for _ in range(5)]
input_sky_path = [[] for _ in range(5)]
input_z = [[] for _ in range(5)]
input_exptime = [[] for _ in range(5)]
input_photflam = [[] for _ in range(5)]
input_photplam = [[] for _ in range(5)]
log_photflam = [[] for _ in range(5)]
log_photplam = [[] for _ in range(5)]
zp_lo = [[] for _ in range(5)]

print("Reading files")

for i in range(len(sci_images_path)):#filtros
    sky_ok = False
    rms_ok = False
    sci_ok = False
    for j in range(len(sci_images_path[i])):#filas

        try:
            with fits.open(sky_images_path[i][j]) as hdul:
                sky_data = hdul[0].data
                sky_ok = True
        except FileNotFoundError:
            print(f"File not found: {sky_images_path[i][j]}, skipping...")
            continue
        try:
            with fits.open(rms_images_path[i][j]) as hdul:
                imerr_data = hdul[0].data
                rms_ok = True
        except FileNotFoundError:
            print(f"File not found: {rms_images_path[i][j]}, skipping...")
            continue
        try:
            with fits.open(sci_images_path[i][j]) as hdul:
                science_data[i].append(hdul[0].data)
                science_header = hdul[0].header
                sci_ok = True
        except FileNotFoundError:
            print(f"File not found: {sci_images_path[i][j]}, skipping...")
            continue



#           -----------Low z parameters-----------
        if sky_ok and rms_ok and sci_ok:
            id_g[i].append(filename[i][j])
            input_image_path[i].append(sci_images_path[i][j])
            input_rms_path[i].append(rms_images_path[i][j])
            input_sky_path[i].append(sky_images_path[i][j])
            input_z[i].append(z[i][j])
            input_exptime[i].append(science_header['EXPTIME'])
            input_photflam[i].append(science_header['PHOTFLAM'])
            input_photplam[i].append(science_header['PHOTPLAM'])
            log_photflam[i].append(np.log10(input_photflam[i][-1]))
            log_photplam[i].append(np.log10(input_photplam[i][-1]))
            zp_lo[i].append(-2.5 * log_photflam[i][-1] - 5.0 * log_photplam[i][-1] - 2.408)


psf_path_list_lo = [  
            r'D:\Documentos\Diego\U\Memoria Titulo\Dopterian\Input\psf\hlsp_clash_hst_wfc3ir-65mas_all_f160w_v1_psf.fits',
            r'D:\Documentos\Diego\U\Memoria Titulo\Dopterian\Input\psf\hlsp_clash_hst_acs-65mas_all_f475w_v1_psf.fits',
            r'D:\Documentos\Diego\U\Memoria Titulo\Dopterian\Input\psf\hlsp_clash_hst_acs-65mas_all_f625w_v1_psf.fits',
            r'D:\Documentos\Diego\U\Memoria Titulo\Dopterian\Input\psf\hlsp_clash_hst_acs-65mas_all_f775w_v1_psf.fits',
            r'D:\Documentos\Diego\U\Memoria Titulo\Dopterian\Input\psf\hlsp_clash_hst_acs-65mas_all_f814w_v1_psf.fits'
        ] 

filter_lo = ['clash_wfc3_f160w','clash_wfc3_f475w','clash_wfc3_f625w','clash_wfc3_f775w','clash_wfc3_f814w']

#low z effective wavelengths
lambda_lo = [15405,4770,6310,7647,8057]

err0_mag = [0.05, 0.05, 0.05, 0.05, 0.05]

# -----------High z parameters----------- 

psf_path_list_hi = [r'D:\Documentos\Diego\U\Memoria Titulo\Dopterian\Input\psf\hlsp_clash_hst_acs-65mas_all_f814w_v1_psf.fits']

filter_hi = ['clash_wfc3_f814w']

#high z effective wavelengths
lambda_hi= 8057


pixscale = 0.065

zp_hi = zp_lo[-1]

print("Creating kcorrect object...")

cos = FlatLambdaCDM(H0=cosmos.H0,Om0=cosmos.Omat,Ob0=cosmos.Obar)
kc = k.kcorrect.Kcorrect(responses=filter_lo, responses_out=filter_hi,responses_map=filter_hi,cosmo=os)

for i in range(len(input_image_path[0])):#por cada galaxia
    print(f"Processing galaxy {id_g[0][i]}")
    lowz_info  = {'redshift': input_z[0][i],
                  'psf': psf_path_list_lo,
                  'zp': [zp_lo[0][i], zp_lo[1][i], zp_lo[2][i], zp_lo[3][i], zp_lo[4][i]],
                  'exptime': [input_exptime[0][i], input_exptime[1][i], input_exptime[2][i], input_exptime[3][i], input_exptime[4][i]],
                  'filter': filter_lo, 
                  'lam_eff': [input_photplam[0][i], input_photplam[1][i], input_photplam[2][i], input_photplam[3][i], input_photplam[4][i]],
                  'pixscale': pixscale,
                  'lambda': lambda_lo}

    highz_info  = {'redshift': 2.0, 
                   'psf': psf_path_list_hi,
                   'zp': zp_hi[i], 
                   'exptime': input_exptime[-1][i],
                   'filter': filter_hi, 
                   'lam_eff': [input_photplam[0][i], input_photplam[1][i], input_photplam[2][i], input_photplam[3][i], input_photplam[4][i]], 
                   'pixscale': pixscale,
                   'lambda': lambda_hi}
    
    

    output_sci = ""
    output_psf = ""
    print(input_image_path[0][i])


    imOUT,psfOUT,n_pkcorrect= dopt.ferengi_k(
                                images = [input_image_path[0][i], input_image_path[1][i], input_image_path[2][i], input_image_path[3][i], input_image_path[4][i]],
                                background= [input_sky_path[0][i], input_sky_path[1][i], input_sky_path[2][i], input_sky_path[3][i], input_sky_path[4][i]],
                                lowz_info = lowz_info, 
                                highz_info = highz_info, 
                                namesout= [output_sci, output_psf], 
                                imerr = [input_rms_path[0][i], input_rms_path[1][i], input_rms_path[2][i], input_rms_path[3][i], input_rms_path[4][i]],
                                err0_mag = err0_mag, 
                                noconv=False, 
                                evo=None, 
                                nonoise=True, 
                                extend=False, 
                                noflux=True,
                                kc_obj=kc)
    if np.any(imOUT != -99):  # Corrección de la condición
        n_images = len(lowz_info['filter']) + 1  # +1 para incluir imOUT

        # Crear la figura y los ejes con subplots dispuestos horizontalmente
        fig, axes = plt.subplots(1, n_images, figsize=(15, 5))  # Cambiado a 1 fila y n_images columnas
        galaxy_name = input_image_lists[0][i]
        galaxy_name = galaxy_name[6:]
        fig.suptitle("Test kcorrect " + galaxy_name + ": " + str(n_pkcorrect) + " pixels corrected", fontsize=16)

        # Mostrar imOUT en el primer subplot
        ax = axes[0]
        im = ax.imshow(imOUT, origin='lower', cmap='gray')
        ax.set_title(f"{highz_info['filter'][0]} z: {highz_info['redshift']}")

        # Crear un eje para la colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax, orientation='vertical')

        # Mostrar las imágenes de input en los subplots restantes
        for j in range(len(lowz_info['filter'])):
            ax = axes[j + 1]
            im = ax.imshow(science_data[j][i], origin='lower', cmap='gray')
            ax.set_title(f"{lowz_info['filter'][j]} z: {lowz_info['redshift']}")
            
            # Crear un eje para la colorbar
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(im, cax=cax, orientation='vertical')

        plt.tight_layout()

        output_dir = r"D:\Documentos\Diego\U\Memoria Titulo\Dopterian\Input\A209\ouput_kcorrect"
        output_filename = "test_kcorrect_" + galaxy_name + ".png"
        output_path = os.path.join(output_dir, output_filename)
        plt.savefig(output_path)
        plt.close(fig)


    
