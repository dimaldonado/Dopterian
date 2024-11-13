#!/usr/bin/env python
# coding: utf-8

# PYTHON PROGRAMME THAT DERIVES THE DIRECT FWHM OF THE PSF
# (Diego Maldonado 06/11/2024)

import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from astropy.cosmology import FlatLambdaCDM


cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Tcmb0=2.725)

#############################################################




def calc_1d_FWHM(psf_integrated, psf_abscissa):

    # determine the maximum (peak) and half maximum PSF flux
    max_intensity = np.max(psf_integrated)
    half_max = max_intensity/2

    # deriving the position of the peak
    max_pos = np.argmax(psf_integrated)

    # selecting the pixels below the half maximum at the left of the peak
    half_pos_left_low = np.where( (psf_abscissa < max_pos) & (psf_integrated < half_max))[0]
    # selecting the pixels above the half maximum at the left of the peak
    half_pos_left_high = np.where( (psf_abscissa < max_pos) & (psf_integrated >= half_max) )[0]
    # interpolating between the two pixels closest to the half maximum
    # -- building the array of x values in which to interpolate
    x_interp_half_max_left = np.arange(half_pos_left_low[-1], half_pos_left_high[0], 0.001)
    # -- defining the boundaries of the interpolation
    xp_left = np.array( [half_pos_left_low[-1], half_pos_left_high[0]] )
    fp_left = np.array( [psf_integrated[half_pos_left_low[-1]], psf_integrated[half_pos_left_high[0]]])
    # -- interpolating
    y_interp_half_max_left = np.interp(x_interp_half_max_left, xp_left, fp_left)
    # -- deriving the interpolated value closest to the halpf maximum
    half_max_pos_left = x_interp_half_max_left[np.argmin(abs(y_interp_half_max_left-half_max))]



    # selecting the pixels below the half maximum at the right of the peak
    half_pos_right_low = np.where( (psf_abscissa > max_pos) & (psf_integrated < half_max))[0]
    # selecting the pixels above the half maximum at the right of the peak
    half_pos_right_high = np.where( (psf_abscissa > max_pos) & (psf_integrated >= half_max) )[0]

    print(half_pos_right_low, half_pos_right_high)
    # interpolating between the two pixels closest to the half maximum
    # -- building the array of x values in which to interpolate
    x_interp_half_max_right = np.arange(half_pos_right_high[-1], half_pos_right_low[0], 0.001)
    # -- defining the boundaries of the interpolation
    xp_right = np.array( [half_pos_right_high[-1], half_pos_right_low[0]] )
    fp_right = np.array( [psf_integrated[half_pos_right_high[-1]], psf_integrated[half_pos_right_low[0]]])
    # -- interpolating
    y_interp_half_max_right = np.interp(x_interp_half_max_right, xp_right, fp_right)

    # -- deriving the interpolated value closest to the halpf maximum
    half_max_pos_right = x_interp_half_max_right[np.argmin(abs(y_interp_half_max_right-half_max))]

    # calculating the FWHM
    fwhm_1d = (half_max_pos_right - half_max_pos_left)

    # returning the output quantity
    return fwhm_1d

####################### END OF calc_1d_FWHM #########################



def calc_FWHM(psf_data, pixel_scale):

    if len(psf_data.shape) == 3 and psf_data.shape[0] ==1:
        psf_data = psf_data.squeeze(axis = 0)


    # obtaining the shape of the PSF data
    psf_shape = psf_data.shape
    print(psf_shape[1])

    # deriving the integrated PSF over the x and y directions
    # and
    # building arrays with the indices of the pixels along the x and y axes (they will be used as abscissas in the estimation of the 1d FWHM)
    
    psf_x = np.sum(psf_data, axis = 0)
    psf_y = np.sum(psf_data, axis = 1)

    psf_indices_x = np.arange(0, psf_shape[1])
    psf_indices_y = np.arange(0, psf_shape[0])

    # estimating the FWHM alng the x and y axes
    fwhm_x = calc_1d_FWHM(psf_x, psf_indices_x)
    fwhm_y = calc_1d_FWHM(psf_y, psf_indices_y)

    # deriving the FWHM
    fwhm = 0.5*(fwhm_x + fwhm_y)* pixel_scale

    # returning the output quantitiy
    return fwhm

####################### END OF calc_FWHM #########################




# FUNCTION THAT RUNS calc_FWHM on a give PSF image
# (Pierluigi Cerulo 07/11/2024)

def run_calc_FWHM(psf_file, pixel_scale, high_z_fwhm, z_low, z_high):

    psf_image_data = fits.getdata(psf_file)

    FWHM = calc_FWHM(psf_image_data, pixel_scale)

    redshifted_FWHM = predict_redshifted_FWHM(FWHM, high_z_fwhm, z_low, z_high)
    
    return FWHM, redshifted_FWHM

####################### END OF run_calc_FWHM #########################



# FUNCTION THAT PREDICTS THE REDSHIFTED PSF FWHM
# (Pierluigi Cerulo 07/11/2024)

def predict_redshifted_FWHM(a_low, high_z_fwhm, z_low, z_high):

    # calcuating the luminosity distances at low and high redshift
    d_low = cosmo.luminosity_distance(z_low) 
    d_high = cosmo.luminosity_distance(z_high)

    # calcuating the predicted FWHM
    a_high = a_low* ( d_low/(1+z_low)**2 )/( d_high/(1+z_high)**2 )

    # comparing the FWHM of the redshifted and high-z PSF
    if a_high > high_z_fwhm:

        print("the FWHM of the redshifted PSF is broader than that of the high-z PSF. The simulation is not possible")

    # returning the predicted high-z FWHM
    return a_high    

####################### END OF run_calc_FWHM #########################



###############################################
#                                             #
#               MAIN PROGRAM                  #
#                                             #
###############################################

# DEFINING INPUT AND OUTPUT FILES

# input
F160W_PSF_file = r'D:\Documentos\Diego\U\Memoria Titulo\Dopterian\Input\psf\hlsp_clash_hst_wfc3ir-65mas_all_f160w_v1_psf.fits'

A383_Subaru_Rc_PSF_file = r'D:\Documentos\Diego\U\Memoria Titulo\Dopterian\Input\psf\A383_Rc_PSFex_psf.fits'

# RUNNING PROCEDURE

F160W_PSF_FWHM , redshifted_F160W_PSF_FWHM = run_calc_FWHM(F160W_PSF_file, 0.065, 0.13, 0.2, 2)
print('F160W_PSF_FWHM', F160W_PSF_FWHM, redshifted_F160W_PSF_FWHM)

A383_Subaru_Rc_PSF_FWHM, redshifted_A383_Subaru_Rc_PSF_FWHM = run_calc_FWHM(A383_Subaru_Rc_PSF_file, 0.2, 0.13, 0.2, 2)
print('A383_Subaru_Rc_PSF_FWHM', A383_Subaru_Rc_PSF_FWHM, redshifted_A383_Subaru_Rc_PSF_FWHM)

####################### END OF PROGRAM #########################
