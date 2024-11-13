import numpy as np
from astropy.io import fits
from dopterian import dopterian as dopt
from dopterian import cosmology as cosmos
from astropy.cosmology import FlatLambdaCDM
import kcorrect as k
import matplotlib.pyplot as plt
import pandas as pd
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable 

#Script para probar la ejecucion de dopterian + kcorrect

catalogs = {
            "B":    r'D:\Documentos\Diego\U\Memoria Titulo\Dopterian\Input\A383\A383_B_input_for_Dopterian.txt',
            "Rc":   r'D:\Documentos\Diego\U\Memoria Titulo\Dopterian\Input\A383\A383_Rc_input_for_Dopterian.txt',
            "Ip":    r'D:\Documentos\Diego\U\Memoria Titulo\Dopterian\Input\A383\A383_Ip_input_for_Dopterian.txt',
            "z":    r'D:\Documentos\Diego\U\Memoria Titulo\Dopterian\Input\A383\A383_z_input_for_Dopterian.txt'
            }
data = {}

# Cargar los datos relevantes
for key, file in catalogs.items():
    data[key] = pd.read_csv(file, delim_whitespace=True, usecols=["CLASHID", "zb_1", "clusterName"])

# Obtener los nombres de los archivos que están presentes en todos los filtros
common_clashids = set(data["B"]["CLASHID"])

for key in data:
    common_clashids.intersection_update(data[key]["CLASHID"])

# Crear las listas de listas
input_image_lists = [[] for _ in range(4)]
z = [[] for _ in range(4)]
clustername = [[] for _ in range(4)]

# Asignar los nombres de los archivos correspondientes a cada filtro, así como zb_1 y clusterName
filters = ["B", "Rc", "Ip", "z"]

for idx, filter in enumerate(filters):
    for clash_id in common_clashids:
        row = data[filter][data[filter]["CLASHID"] == clash_id]
        if not row.empty:
            input_image_lists[idx].append(f"{filter}_{row['CLASHID'].values[0]}.fits")
            z[idx].append(row["zb_1"].values[0])
            clustername[idx].append(row["clusterName"].values[0])

base_path = "D:\\Documentos\\Diego\\U\\Memoria Titulo\\Dopterian\\Input\\A383\\"


filename = input_image_lists
sci_images_path = [[base_path+"SCI_" + image for image in sublist] for sublist in input_image_lists]
rms_images_path = [[base_path+"RMS_" + image for image in sublist] for sublist in input_image_lists]
sky_images_path = [[base_path+"sky_" + image for image in sublist] for sublist in input_image_lists]


science_data = [[] for _ in range(4)]
id_g= [[] for _ in range(4)]
input_image_path = [[] for _ in range(4)]
input_rms_path = [[] for _ in range(4)]
input_sky_path = [[] for _ in range(4)]
input_z = [[] for _ in range(4)]


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
            


psf_path_list_lo = [  
            r'D:\Documentos\Diego\U\Memoria Titulo\Dopterian\Input\psf\A383_B_PSFex_psf.fits',
            r'D:\Documentos\Diego\U\Memoria Titulo\Dopterian\Input\psf\A383_Rc_PSFex_psf.fits',
            r'D:\Documentos\Diego\U\Memoria Titulo\Dopterian\Input\psf\A383_Ip_PSFex_psf.fits',
            r'D:\Documentos\Diego\U\Memoria Titulo\Dopterian\Input\psf\A383_Z_PSFex_psf.fits'
            ] 

filter_lo = ['subaru_suprimecam_B','subaru_suprimecam_Rc','subaru_suprimecam_i','subaru_suprimecam_z']

input_exptime = [8400,13800,2400,4500]

zp_lo = [23.972,24.089,24.25,23.708]

pixscale_lo = 0.2

#low z effective wavelengths
lambda_lo = [4400,6500,7700,9000]

err0_mag = [0.05, 0.05, 0.05, 0.05]

# -----------High z parameters----------- 

psf_path_list_hi = r'D:\Documentos\Diego\U\Memoria Titulo\Dopterian\Input\psf\hlsp_clash_hst_wfc3ir-65mas_all_f160w_v1_psf.fits'

filter_hi = ['clash_wfc3_f160w']

exptime_hi = 5935.212

#high z effective wavelengths
lambda_hi= 15405

pixscale_hi = 0.065

zp_hi = 25.946223685088732


responses_map = ['subaru_suprimecam_z']
print(len(input_image_path[0]))

print("Creating kcorrect object...")

cos = FlatLambdaCDM(H0=cosmos.H0,Om0=cosmos.Omat,Ob0=cosmos.Obar)
kc = k.kcorrect.Kcorrect(responses=filter_lo, responses_out=filter_hi,responses_map=responses_map,cosmo=os)

for i in range(len(input_image_path[0])):#por cada galaxia

    name = id_g[0][i]
    name = name.replace("F160W_", "")
    print(f"Processing galaxy {name}")

    lowz_info  = {'redshift': input_z[0][i],
                  'psf': psf_path_list_lo,
                  'zp': [zp_lo[0], zp_lo[1], zp_lo[2], zp_lo[3]],
                  'exptime': [input_exptime[0], input_exptime[1], input_exptime[2], input_exptime[3]],
                  'filter': filter_lo, 
                  'pixscale': pixscale_lo,
                  'lambda': lambda_lo}

    highz_info  = {'redshift': 2.0, 
                   'psf': psf_path_list_hi,
                   'zp': zp_hi, 
                   'exptime': exptime_hi,
                   'filter': filter_hi, 
                   'pixscale': pixscale_hi,
                   'lambda': lambda_hi}
    
    
    output_sci = "D:\\Documentos\\Diego\\U\\Memoria Titulo\Dopterian\\Input\\A383\\ouput_kcorrect\\fits\\"+"output_sci_"+name
    output_psf = "D:\\Documentos\\Diego\\U\\Memoria Titulo\Dopterian\\Input\\A383\\ouput_kcorrect\\fits\\"+"output_psf_"+name
    print(input_image_path[0][i])
    print(input_image_path[1][i])
    print(input_image_path[2][i])
    print(input_image_path[3][i])
    imOUT,psfOUT,n_pkcorrect= dopt.ferengi(
                                images = [input_image_path[0][i], input_image_path[1][i], input_image_path[2][i], input_image_path[3][i]],
                                background= [input_sky_path[0][i], input_sky_path[1][i], input_sky_path[2][i], input_sky_path[3][i]],
                                lowz_info = lowz_info, 
                                highz_info = highz_info, 
                                namesout= [output_sci, output_psf], 
                                imerr = [input_rms_path[0][i], input_rms_path[1][i], input_rms_path[2][i], input_rms_path[3][i]],
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

        output_dir = r"D:\Documentos\Diego\U\Memoria Titulo\Dopterian\Input\A383\ouput_kcorrect"
        output_filename = "test_kcorrect_" + galaxy_name + ".png"
        output_path = os.path.join(output_dir, output_filename)
        plt.savefig(output_path)
        plt.close(fig)

