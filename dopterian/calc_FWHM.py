#!/usr/bin/env python
# coding: utf-8

# PYTHON PROGRAMME THAT DERIVES THE DIRECT FWHM OF THE PSF
# (Diego Maldonado 06/11/2024)

import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit




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

