import numpy as np
import scipy.ndimage as scndi
import astropy.io.fits as pyfits
import astropy.convolution as apcon
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import statsmodels.api as sm
import scipy.integrate as scint 
import numpy.random as npr
import astropy.io.fits as pyfits
import scipy.optimize as scopt
import astropy.modeling as apmodel
import warnings
import scipy.ndimage as scndi
from dopterian import cosmology as cosmos
import astropy.convolution as apcon
from astropy.cosmology import FlatLambdaCDM
from dopterian import cosmology as cosmos
import kcorrect
from scipy.optimize import curve_fit
from dopterian import calc_FWHM
import os

version = '1.0.0'   


version = '1.0.0'   
c = 299792458. ## speed of light


## SDSS maggies to lupton
magToLup = {'u':1.4e-10,'g':0.9e-10,'r':1.2e-10,'i':1.8e-10,'z':7.4e-10}

#==============================================================================
# HELPER FUNCTIONS
#==============================================================================

def nu2lam(nu):
    "From Hz to Angstrom"
    return c/nu*1e-10
    
def lam2nu(lam):
    "From Angstrom to Hz"
    return c/lam*1e10
    
def maggies2mags(maggies):
    return -2.5*np.log10(maggies)

def mags2maggies(mags):
    return 10**(-0.4*mags)

def maggies2fnu(maggies):
    "From maggies to units of [erg s-1 Hz-1 cm-2]"
    return 3531e-23*maggies 

def fnu2maggies(fnu):
    "From [erg s-1 Hz-1 cm-2] to maggies"
    return 3631e23*fnu

def fnu2flam(fnu,lam):
    return c*1e10/lam**2*fnu
    
def flam2fnu(flam,lam):
    return flam/c/1.0e10*lam**2

def lambda_eff(lam,trans):
    "Calculate the mean wavelength of a filter"
    indexs = np.where(lam != 0)[0]
    if len(indexs)==0:
        raise ValueError('ERROR: no non-zero wavelengths')
    else:
        Lambda=np.squeeze(lam[indexs])
        Transmission=np.squeeze(trans[indexs])
        return scint.simps(Lambda*Transmission,Lambda)/scint.simps(Transmission,Lambda)

def cts2maggies(cts,exptime,zp):
    return cts/exptime*10**(-0.4*zp)
    
def cts2mags(cts,exptime,zp):
    return maggies2mags(cts2maggies(cts,exptime,zp))

def maggies2cts(maggies,exptime,zp):
    return maggies*exptime/10**(-0.4*zp)
    
def mags2cts(mags,exptime,zp):
    return maggies2cts(mags2maggies(mags),exptime,zp)


def maggies2lup(maggies,filtro):
    b = magToLup[filtro]
    return -2.5/np.log(10)*(np.arcsinh(maggies/b*0.5)+np.log(b))
    
def lup2maggies(lup,filtro):
#    maggies=lup
    b = magToLup[filtro]
    return 2*b*np.sinh(-0.4*np.log(10)*lup-np.log(b))
    
def random_indices(size,indexs):
    "Returns an array of a set of indices from indexs with a size with no duplicates."
    return npr.choice(indexs,size=size,replace=False)

def edge_index(a,rx,ry):
    "The routine creates an index list of a ring with width 1 around the centre at radius rx and ry"
    N,M=a.shape
    XX,YY=np.meshgrid(np.arange(N),np.arange(M))
    
    Y = np.abs(XX-N/2.0).astype(np.int64)
    X = np.abs(YY-M/2.0).astype(np.int64)
    
    idx = np.where(((X==rx) * (Y<=ry)) + ((Y==ry) * (X<=rx)))
####    CHECK
    return idx


def predict_redshifted_FWHM(a_low, z_low, z_high):

    # calcuating the luminosity distances at low and high redshift

    d_low = cosmos.luminosity_distance(z_low) 
    d_high = cosmos.luminosity_distance(z_high)

    # calcuating the predicted FWHM
    a_high = a_low* ( d_low/(1+z_low)**2 )/( d_high/(1+z_high)**2 )

    
    # returning the predicted high-z FWHM
    return a_high    

def check_PSF_FWHM(psf_low,psf_high,pixcale_low_z,pixscale_high_z,z_low,z_high):

    low_z_fwhm = calc_FWHM.calc_FWHM(psf_low, pixcale_low_z)
    high_z_fwhm = calc_FWHM.calc_FWHM(psf_high, pixscale_high_z)

    redshifted_low_z_fwhm = predict_redshifted_FWHM(low_z_fwhm, z_low, z_high)

    if redshifted_low_z_fwhm > high_z_fwhm:
        return False
    
    return True
    
    
        



def dist_ellipse(img,xc,yc,q,ang):
    "Compute distance to the center xc,yc in elliptical apertures. Angle in degrees."
    ang=np.radians(ang)

    X,Y = np.meshgrid(range(img.shape[1]),range(int(img.shape[0])))
    rX=(X-xc)*np.cos(ang)-(Y-yc)*np.sin(ang)
    rY=(X-xc)*np.sin(ang)+(Y-yc)*np.cos(ang)
    dmat = np.sqrt(rX*rX+(1/(q*q))*rY*rY)
    return dmat

def robust_linefit(x, y):
    print(x)
    print(y)
    X = sm.add_constant(x)  # Agrega una constante para el término independiente
    robust_model = sm.RLM(y, X, M=sm.robust.norms.HuberT())
    results = robust_model.fit()
    return results.params

def resistent_mean(a,k):
    """Compute the mean value of an array using a k-sigma clipping method
    """
    a = np.asanyarray(a)
    media=np.nanmean(a)
    dev=np.nanstd(a)

    back=a.copy()
    back=back[back!=0]
    thresh=media+k*dev
    npix = len(a[a>=thresh])
    while npix>0:
        back = back[back<thresh]
        media=np.mean(back)
        dev=np.std(back)
        thresh=media+k*dev
        npix = len(back[back>=thresh])
        
    nrej = np.size(a[a>=thresh])
    return media,dev,nrej

def ring_sky(image,width0,nap,x=None,y=None,q=1,pa=0,rstart=None,nw=None):
    """ For an image measure the flux around postion x,y in rings with
    axis ratio q and position angle pa with nap apertures (for sky slope). rstart indicates 
    the starting radius and nw (if set) limits the width to the number of apertures
    and not calculated in pixels.
    """
    
    if nap<=3:
        raise ValueError('Number of apertures must be greater than 3.')

    if type(image)==str:
        image=pyfits.getdata(image)
    
    N,M=image.shape
    if rstart is None:
        rstart=0.05*min(N,M)
    
    if x is None and y is None:
        x=N*0.5
        y=M*0.5
    elif x is None or y is None:
        raise ValueError('X and Y must both be set to a value')
    else:
        pass
    
    rad = dist_ellipse(image,x,y,q,pa)
    max_rad=0.95*np.amax(rad)
    
    if nw is None:
        width=width0
    else:
        width=max_rad/float(width0)
    
    media,sig,nrej=resistent_mean(image,3)
    sig*=np.sqrt(np.size(image)-1-nrej)
    
    if rstart is None:
        rhi=width
    else:
        rhi=rstart
    
    nmeasures=2
    
    r=np.array([])
    flux=np.array([])
    i=0
    while rhi<=max_rad:
        extra=0
        ct=0
        while ct<10:
            idx = (rad<=rhi+extra)*(rad>=rhi-extra)*(np.abs(image)<3*sig)
            ct=np.size(image[idx])
            
            extra+=1
            if extra>max(N,M)*2:
                break
            
        if ct<5:
            sky = flux[len(flux)-1]
        else:
            sky = resistent_mean(image[idx],3)[0]
        
        r=np.append(r,rhi-0.5*width)
        flux=np.append(flux,sky)
        
        i+=1
        if np.size(flux) > nap:
            # Filtrar valores inf y NaN
            valid_indices = np.isfinite(r[i-nap+1:i]) & np.isfinite(flux[i-nap+1:i])
            if np.sum(valid_indices) > 0:
                pars, err = scopt.curve_fit(lambda x, a, b: a * x + b, r[i-nap+1:i][valid_indices], flux[i-nap+1:i][valid_indices])
                slope = pars[0]
                if slope > 0 and nmeasures == 0:
                    break
                elif slope > 0:
                    nmeasures -= 1

        rhi += width
    sky = resistent_mean(flux[i-nap+1:i],3)[0]        
    return sky

def ferengi_make_psf_same(psf1,psf2):
    "Compares the size of both psf images and zero-pads the smallest one so that they have the same size"

    if np.size(psf1)>np.size(psf2):
        case=True
        big=psf1
        small=psf2
    else:
        big=psf2
        small=psf1
        case=False

    Nb,Mb=big.shape
    Ns,Ms=small.shape
    
    center = int(np.floor(Nb/2))#revisar
    small_side = int(np.floor(Ns/2))#revisar
    new_small=np.zeros(big.shape)
    new_small[center-small_side:center+small_side+1,center-small_side:center+small_side+1]=small
    if case==True:
        return psf1,new_small
    else:
        return new_small,psf2

def barycenter(img,segmap):
    """ Compute the barycenter of a galaxy from the image and the segemntation map.
    """
    N,M=img.shape
    XX,YY=np.meshgrid(range(M),range(N))
    gal=abs(img*segmap)
    Y = np.average(XX,weights=gal)
    X = np.average(YY,weights=gal)     
    return X,Y


def ferengi_psf_centre(psf,debug=False):
    # Verificamos si el psf tiene una tercera dimensión que sea de tamaño 1
    if len(psf.shape) == 3 and psf.shape[0] == 1:
        psf = psf.squeeze(axis=0)  # Elimina la primera dimensión si es 1
    "Center the psf image using its light barycenter and not a 2D gaussian fit."
    N,M=psf.shape
    
    assert N==M,'PSF image must be square'
    
    if N%2==0:
        center_psf=np.zeros([N+1,M+1])
    else:
        center_psf=np.zeros([N,M])

##    if debug:
##        print np.amax(psf),np.amin(psf)
##        mpl.imshow(psf);mpl.show()
        

    center_psf[0:N,0:M]=psf
    N1,M1=center_psf.shape

    X,Y=barycenter(center_psf,np.ones(center_psf.shape))
    G2D_model = apmodel.models.Gaussian2D(np.amax(psf),X,Y,3,3)

    fit_data = apmodel.fitting.LevMarLSQFitter()
    X,Y=np.meshgrid(np.arange(N1),np.arange(M1))
    with warnings.catch_warnings(record=True) as w:
        pars = fit_data(G2D_model, X, Y, center_psf)

    if len(w)==0:
        cenY = pars.x_mean.value
        cenX = pars.y_mean.value
    else:
        for warn in w:
            print(warn)
        cenX = center_psf.shape[0]/2+N%2
        cenY = center_psf.shape[1]/2+N%2
    
    dx =(cenX-center_psf.shape[0]/2)
    dy =(cenY-center_psf.shape[1]/2)
    
    center_psf= scndi.shift(center_psf,[-dx,-dy])
    
    
    return center_psf#,pars
    

def ferengi_deconvolve(wide,narrow):#TBD
    "Images should have the same size. PSFs must be centered (odd pixel numbers) and normalized."

    Nn,Mn=narrow.shape #Assumes narrow and wide have the same shape

    smax = max(Nn,Mn) 
    bigsz=2    
    while bigsz<smax:
        bigsz*=2

    if bigsz>2048:
        print('Requested PSF array is larger than 2x2k!')
    
    psf_n_2k = np.zeros([bigsz,bigsz],dtype=np.double)
    psf_w_2k = np.zeros([bigsz,bigsz],dtype=np.double)
    
    psf_n_2k[0:Nn,0:Mn]=narrow
    psf_w_2k[0:Nn,0:Mn]=wide
    
#    fig,ax=mpl.subplots(1,2,sharex=True,sharey=True)
#    ax[0].imshow(psf_n_2k)
#    ax[1].imshow(psf_w_2k)
#    mpl.show()

    psf_n_2k=psf_n_2k.astype(np.complex_)
    psf_w_2k=psf_w_2k.astype(np.complex_)
    fft_n = np.fft.fft2(psf_n_2k)
    fft_w = np.fft.fft2(psf_w_2k)
    
    fft_n = np.absolute(fft_n)/(np.absolute(fft_n)+0.000000001)*fft_n
    fft_w = np.absolute(fft_w)/(np.absolute(fft_w)+0.000000001)*fft_w
    
    psf_ratio = fft_w/fft_n

    
#    Create Transformation PSF
    psf_intermed = np.real(np.fft.fft2(psf_ratio))
    psf_corr = np.zeros(narrow.shape,dtype=np.double)
    lo = bigsz-Nn//2
    hi=Nn//2
    psf_corr[0:hi,0:hi]=psf_intermed[lo:bigsz,lo:bigsz]    
    psf_corr[hi:Nn-1,0:hi]=psf_intermed[0:hi,lo:bigsz]    
    psf_corr[hi:Nn-1,hi:Nn-1]=psf_intermed[0:hi,0:hi]    
    psf_corr[0:hi,hi:Nn-1]=psf_intermed[lo:bigsz,0:hi]        
    
    
    psf_corr = np.rot90(psf_corr,2)

    '''
    plt.figure()
    plt.imshow(psf_corr/np.sum(psf_corr), cmap='gray')
    plt.title('psf_corr')
    plt.colorbar()

    plt.figure()
    plt.imshow(ferengi_psf_centre(psf_corr/np.sum(psf_corr)), cmap='gray')
    plt.title('psf_corr_centre')
    plt.colorbar()
    plt.show()
    '''
    
    return psf_corr/np.sum(psf_corr)
    
def ferengi_clip_edge(image,auto_frac=2,clip_also=None,norm=False):#TBD
    N,M=image.shape
    rx = int(N/2/auto_frac)
    ry = int(M/2/auto_frac)
    
    sig=np.array([])
    r=np.array([])
    while True:
        idx = edge_index(image,rx,ry)
        if np.size(idx[0])==0:
            break
        med,sigma,nrej=resistent_mean(image,3)
        sigma*=np.sqrt(np.size(image)-1-nrej)
        sig=np.append(sig,sigma)
        r=np.append(r,rx)
        rx+=1
        ry+=1
    
    new_med,new_sig,new_nrej=resistent_mean(sig,3)
    new_sig*=np.sqrt(np.size(sig)-1-new_nrej)
    
    i=np.where(sig>=new_med*10*new_sig)
    if np.size(i)>0:
        lim = np.min(r[i])
        if np.size(i)>new_nrej*3:
            print('Large gap?')
        npix = round(N/2.0-lim)
        
        if clip_also is not None:
            clip_also = clip_also[npix:N-1-npix,npix:M-1-npix]
        image=image[npix:N-1-npix,npix:M-1-npix]
    
    if norm==True:
        image/=np.sum(image)
        if clip_also is not None:
            clip_also/=np.sum(clip_also)
    
    if clip_also is not None:
        return npix,image,clip_also
    else:
        return npix,image

def rebin2d(img,Nout,Mout,flux_scale=False):
    """Special case of non-integer magnification for 2D arrays
    from FREBIN of IDL Astrolib.
    """

    N,M = img.shape

    xbox = N/float(Nout)
    ybox = M/float(Mout)

    temp_y = np.zeros([N,Mout])

    for i in range(Mout):
        rstart = i*ybox
        istart = int(rstart)

        rstop = rstart + ybox
        if int(rstop) > M-1:
            istop = M-1
        else:
            istop = int(rstop)

        frac1 = rstart-istart
        frac2 = 1.0 - (rstop-istop)       
        if istart == istop:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                temp_y[:, i] = (1.0 - frac1 - frac2) * img[:, istart]
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                temp_y[:, i] = np.sum(img[:, istart:istop + 1], 1) - frac1 * img[:, istart] - frac2 * img[:, istop]

    temp_y = temp_y.transpose()
    img_bin = np.zeros([Mout,Nout])

    for i in range(Nout):
        rstart = i*xbox
        istart = int(rstart)

        rstop = rstart + xbox
        if int(rstop) > N-1:
            istop = N-1
        else:
            istop = int(rstop)

        frac1 = rstart-istart
        frac2 = 1.0 - (rstop-istop)

        if istart == istop:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                img_bin[:, i] = (1.0 - frac1 - frac2) * temp_y[:, istart]
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                img_bin[:, i] = np.sum(temp_y[:, istart:istop + 1], 1) - frac1 * temp_y[:, istart] - frac2 * temp_y[:, istop]      

    if flux_scale:
        return img_bin.transpose()
    else:
        return img_bin.transpose()/(xbox*ybox)
        
        
def lum_evolution(zlow,zhigh):
    "Defined Luminosity evolution from L* of Sobral et al. 2013."
    def luminosity(z):
        logL = 0.45*z+41.87
        return 10**(logL)
    return luminosity(zhigh)/luminosity(zlow)

def ferengi_downscale(image_low,z_low,z_high,pix_low,pix_hi,upscale=False,nofluxscale=False,evo=None):
    da_in = cosmos.angular_distance(z_low)
    da_out = cosmos.angular_distance(z=z_high)

    
    dl_in=cosmos.luminosity_distance(z_low)
    dl_out=cosmos.luminosity_distance(z_high)
   
    
    if evo is not None:
        evo_fact = 10 ** (-0.4 * evo * z_high) ### UPDATED TO MATCH FERENGI ALGORITHM numeros negativos entre 0 y -1
    else:
        evo_fact = lum_evolution(z_low,z_high)
        #evo_fact = 1.0
    mag_factor = (da_in/da_out)*(pix_low/pix_hi)
    if upscale == True:
        mag_factor=1.0/mag_factor
    
##    lum_factor = (dl_in/dl_out)**2
    lum_factor = (dl_in/dl_out)**2*(1.+z_high)/(1.+z_low) ### UPDATED TO MATCH FERENGI ALGORITHM
        
    if nofluxscale==True:
        lum_factor=1.0
    else:
        lum_factor =(da_in/da_out)**2
        
    N,M = image_low.shape
    
    N_out = int(round(N * mag_factor))
    M_out = int(round(M * mag_factor))

    img_out = rebin2d(image_low,N_out,M_out,flux_scale=True)*lum_factor*evo_fact

    return img_out


def ferengi_odd_n_square():
    #TBD : in principle avoidable if PSF already square image
    #feregi_psf_centre already includes number of odd pixels
    raise NotImplementedError('In principle avoidable if PSF already square image')
    return


def ferengi_transformation_psf(psf_low,psf_high,z_low,z_high,pix_low,pix_high,same_size=None):
    """ Compute the transformation psf. Psf_low and psf_high are the low and high redshift PSFs respectively.
    Also needed as input paramenters the redshifts (low and high) and pixelscales (low and high).
    """    

    psf_l = ferengi_psf_centre(psf_low)
    psf_h = ferengi_psf_centre(psf_high)

    da_in = cosmos.angular_distance(z_low)
    da_out = cosmos.angular_distance(z_high)

    N,M=psf_l.shape
    add=0
    out_size = round((da_in/da_out)*(pix_low/pix_high)*(N+add))
    
    
    while out_size%2==0:
        add+=2
        psf_l=np.pad(psf_l,1,mode='constant')
        out_size = round((da_in/da_out)*(pix_low/pix_high)*(N+add))
        if add>N*3:
            return -99
##            raise ValueError('Enlarging PSF failed!')
    
   
    psf_l = ferengi_downscale(psf_l,z_low,z_high,pix_low,pix_high,nofluxscale=True)
    psf_l = ferengi_psf_centre(psf_l)

# Make the psfs the same size (then center)
    psf_l,psf_h=ferengi_make_psf_same(psf_l,psf_h)  
    psf_l = ferengi_psf_centre(psf_l)
    psf_h = ferengi_psf_centre(psf_h)
    
    
# NORMALIZATION    
    psf_l/=np.sum(psf_l)
    psf_h/=np.sum(psf_h)
    

    return psf_l,psf_h,ferengi_psf_centre(ferengi_deconvolve(psf_h,psf_l))

def ferengi_convolve_plus_noise(image,psf,sky,exptime,nonoise=False,border_clip=None,extend=False):
    
    if border_clip is not None: ## CLIP THE PSF BORDERS (CHANGE IN INPUT PSF)
        Npsf,Mpsf=psf.shape
        psf = psf[border_clip:Npsf-border_clip,border_clip:Mpsf-border_clip]
    
    Npsf,Mpsf=psf.shape
    Nimg,Mimg=image.shape

    out = np.pad(image,Npsf,mode='constant') #enlarging the image for convolution by zero-padding the psf image size
    
    out = apcon.convolve_fft(out,psf/np.sum(psf))
    
    Nout,Mout=out.shape
    if extend==False:
        out=out[Npsf:Nout-Npsf,Mpsf:Mout-Mpsf]
        
    Nout,Mout=out.shape # grab new dimensions, if escaped
    
    Nsky,Msky=sky.shape

    if nonoise==False:
        ef= Nout%2
        try:
            out+=sky[Nsky//2-Nout//2:Nsky//2+Nout//2+ef,Msky//2-Mout//2:Msky//2+Mout//2+ef]+np.sqrt(np.abs(out*exptime))*npr.normal(size=out.shape)/exptime
        except ValueError:
            return -99*np.ones(out.shape)
    else: 
        out = out
    return out

import os


def dump_results(image, psf, imgname_in, bgimage_in, names_out, lowz_info, highz_info):
    name_imout, name_psfout = names_out
    
    Hprim = pyfits.PrimaryHDU(data=image)
    hdu = pyfits.HDUList([Hprim])
    hdr_img = hdu[0].header
    
    # Utility function to add multi-entry parameters as separate numbered columns
    def add_multi_entry(header, base_key, values, description, enumerate_single=False):
        if isinstance(values, list):
            if len(values) == 1 and not enumerate_single:  # No enumeration if only one element and enumerate_single is False
                header[base_key] = (str(values[0]), description)
            else:
                for i, value in enumerate(values, 1):
                    # Shorten base_key to 6 characters to keep within 8-character limit when adding "_i" suffix
                    header[f"{base_key[:6]}_{i}"] = (str(value), f"{description} {i}")
        else:
            header[base_key] = (str(values), description)

    # Convert input image names and background image names to filenames only
    imgname_list = [os.path.basename(str(path)) for path in imgname_in] if isinstance(imgname_in, list) else [os.path.basename(str(imgname_in))]
    bgimage_list = [os.path.basename(str(path)) for path in bgimage_in] if isinstance(bgimage_in, list) else [os.path.basename(str(bgimage_in))]

    # Add each entry as separate columns
    add_multi_entry(hdr_img, 'INPUT', imgname_list, 'Input image')
    add_multi_entry(hdr_img, 'SKYIM', bgimage_list, 'Background image')  # Changed 'SKY_IMG' to 'SKYIM'

    # Process low-z information with multi-column handling and enumeration for all lists
    for key, value in lowz_info.items():
        base_key = f"{key[:4].upper()}_I"
        if key.lower() == "psf":
            value_list = [os.path.basename(str(v)) for v in value] if isinstance(value, list) else [os.path.basename(str(value))]
        else:
            value_list = value if isinstance(value, list) else [value]
        add_multi_entry(hdr_img, base_key, value_list, f"{key} input lowz", enumerate_single=True)

    hdr_img['comment'] = f'Using ferengi.py version {version}'

    # Process high-z information, avoiding enumeration if there’s a single value in lists
    for key, value in highz_info.items():
        base_key = f"{key[:4].upper()}_O"
        if key.lower() == "psf":
            value_list = [os.path.basename(str(v)) for v in value] if isinstance(value, list) else [os.path.basename(str(value))]
        else:
            value_list = value if isinstance(value, list) else [value]
        # For high-z info, set enumerate_single=False to avoid suffix if there's a single value
        add_multi_entry(hdr_img, base_key, value_list, f"{key} input highz", enumerate_single=False)
    
    # Write the image and PSF to files
    hdu.writeto(name_imout, overwrite=True)
    pyfits.writeto(name_psfout, psf, overwrite=True)
    return

def kcorrect_maggies(image,im_err,lowz_info,highz_info,lambda_lo,lambda_hi,err0_mag=None,evo=None,noflux=None,kc_obj=None,):
        n_bands = len(image)
        dz = np.abs(lambda_hi / lambda_lo - 1)
        idx_bestfilt = np.argmin(dz)
        
    #weight the closest filters in rest-frame more
        dz1 = np.abs(lambda_hi - lambda_lo)
        ord = np.argsort(dz1)#Indices de los valores ordenados de menor a mayor
        weight = np.ones(n_bands)#[1,1,1,1,1]
        if dz1[ord[0]] == 0:  
            if n_bands == 2:
                weight[ord] = [10, 4]
            elif n_bands == 3:
                weight[ord] = [10, 4, 4]
            elif n_bands >= 4:
                weight[ord] = [10, 4, 4] + [1] * (n_bands - 3)
                             
        else:
            if n_bands == 2:
                weight[ord] = [10, 8]
            elif n_bands == 3 or n_bands == 4:
                weight[ord] = [10, 8] + [4] * (n_bands - 2)
            elif n_bands > 4:
                weight[ord] = [10, 8, 4, 4] + [1] * (n_bands - 4)
        
    
        img_downscale = []
        imerr_downscale = []

        for i in range(n_bands):
            img_downscale.append(ferengi_downscale(image[i],lowz_info['redshift'],highz_info['redshift'],lowz_info['pixscale'],highz_info['pixscale'],evo=evo,nofluxscale=noflux))
            
            img_downscale[i]-=ring_sky(img_downscale[i], 50,15,nw=True)
        
            imerr_downscale.append(ferengi_downscale(im_err[i],lowz_info['redshift'],highz_info['redshift'],lowz_info['pixscale'],highz_info['pixscale'],evo=evo,nofluxscale=noflux))

            #convert the error from cts to mags
            imerr_downscale[i] = 2.5 / np.log(10) * imerr_downscale[i] / img_downscale[i]

            #calculate the flux in each pixel (convert image from cts to maggies)
            img_downscale[i] = cts2maggies(img_downscale[i],lowz_info['exptime'][i],lowz_info['zp'][i])
            
        
        
        
        #siglim defines the sigma above which K-corrections are calculated
        siglim = 2
        sig = np.zeros(n_bands) # [0,0,0,0,0]

        npix = np.size(img_downscale[0])#numero pizeles de cada banda 28x28

        zmin = np.abs(lambda_hi / lambda_lo - 1 - highz_info['redshift'])
        filt_i = np.argmin(zmin)#indice del menor valor

        nsig = np.zeros_like(img_downscale) #[[28x28],[28x28],[28x28],[28x28],[28x28]]
        nhi = np.zeros_like(img_downscale[0])#[28x28]
        
        #select the pixels above nsig with resistant_mean
        #create a sigma map
        for i in range(n_bands):
            m, s, n = resistent_mean(img_downscale[i], 3)
            sig[i] = s * np.sqrt(npix - 1 - n)# [0,0,0,0,0]
            nsig[i] = scndi.median_filter(img_downscale[i],size=3) / sig[i]#para cada pixel de todas las bandas
            hi = np.where(np.abs(nsig[i]) > siglim)# para cada pixel de todas las bandas
            if hi[0].size > 0:                     # se cuenta el numero de veces
                nhi[hi] += 1                       # que el valor de nsig es mayor a siglim
                                                   # y se almacena su indice en hi 
            
         
        #from the "closest" filter select good pixels
        good1 = np.where((np.abs(nsig[filt_i]) > 0.25) & (np.abs(nsig[filt_i]) <= siglim))

        #select only 50% of all pixels with 0.25 < nsig < siglim
        if good1[0].size > 0:
            n_selec = round(good1[0].size * 0.5)
            good1_indices = np.random.choice(good1[0].size, size=n_selec, replace=False)
            good1 = (good1[0][good1_indices], good1[1][good1_indices])
        
        good = np.where((nhi >= 3) & (np.abs(nsig[filt_i]) > siglim))
        if good[0].size > 0:
            print('3+ filters have high sigma pixels')
        else:
            print('Less than 3 filters have high sigma pixels')
            good = np.where((nhi >= 2) & (np.abs(nsig[filt_i]) > siglim))
            if good[0].size == 0:
                print('Less than 2 filters have high sigma pixels')
                good = np.where((nhi >= 1) & (np.abs(nsig[filt_i]) > siglim))
                if good[0].size == 0:
                    print('NO filter has high sigma pixels')
                    good = np.where((nhi >= 0) & (np.abs(nsig[filt_i]) > siglim))
        
        #se concatenan los indices de los pixeles que cumplen con la condicion                    
        if good1[0].size > 0:
            good = (np.concatenate((good[0], good1[0])), np.concatenate((good[1], good1[1])))
            combined_indices = np.vstack((good[0], good[1])).T
            unique_indices = np.unique(combined_indices, axis=0)#se eliminan los indices repetidos
            good = (unique_indices[:, 0], unique_indices[:, 1])
        
        

        ngood = good[0].size
        if ngood == 0:
            print('No pixels to process')
        else: 
            print(str(ngood) + ' pixels to process')
                
        
            maggies = []
            err = []
            nsig_2d = [] 

            #setup the arrays for the pixels that are to be K-corrected
            for i in range(ngood):
                aux_maggies = []
                aux_err = []
                aux_nsig = []
                for j in range(n_bands):
                    aux_maggies.append(img_downscale[j][good[0][i]][good[1][i]])
                    aux_err.append(imerr_downscale[j][good[0][i]][good[1][i]])  
                    aux_nsig.append(nsig[j][good[0][i]][good[1][i]])
                maggies.append(np.array(aux_maggies))
                nsig_2d.append(np.array(aux_nsig))
                err.append(np.array(aux_err))

            #remove infinite values in the error image
            for i in range(ngood):
                inf = np.where(~np.isfinite(err[i]))
                if inf[0].size > 0:
                    err[i][inf] = 99999
                err[i] = np.abs(err[i])
                err[i] = np.where(err[i] < 99999, err[i], 99999)
            
            # Setup array with minimum errors for SDSS

            err0 = np.tile(err0_mag, (ngood, 1)) #crea matriz con los valores de err0_mag en cada fila ngood veces
            wei = np.tile(weight, (ngood, 1))    #crea matriz con los valores de weight en cada fila ngood veces

            #add image errors and minimum errors in quadrature
            err = np.array(err)
            err = np.sqrt(err0**2 + err**2)/wei
            
            # Convert errors from magnitudes to ivar (inverse variance)
            ivar = (2.5 / np.log(10) / err / maggies)**2

            inf = np.where(~np.isfinite(ivar))
            if inf[0].size > 0:
                ivar[inf] = np.max(ivar[np.isfinite(ivar)])

            responses_lo = lowz_info['filter']
            responses_hi = highz_info['filter']
            redshift_lo = lowz_info['redshift']*np.ones(ngood)
            redshift_hi = highz_info['redshift']*np.ones(ngood)

            if kc_obj is not None:
                
                kc = kc_obj
            else:
                #kcorrect object 
                print("Creating kcorrect object...")
                cos = FlatLambdaCDM(H0=cosmos.H0,Om0=cosmos.Omat,Ob0=cosmos.Obar)
                kc = kcorrect.kcorrect.Kcorrect(responses=responses_lo,responses_out=[responses_hi],responses_map=[responses_lo[idx_bestfilt]],cosmo=cos)
            
            coeffs = kc.fit_coeffs(redshift = redshift_lo,maggies = maggies,ivar = ivar)
            k_values =  kc.kcorrect(redshift=redshift_lo, coeffs=coeffs)
            
            #reconstruct magnitudes in a certain filter at a certain redshift
            r_maggies = kc.reconstruct_out(redshift=redshift_hi,coeffs=coeffs)

        #as background choose closest in redshift-space
       
        bg = img_downscale[filt_i] / (1.0 + highz_info['redshift'])
        img_downscale = bg
        #put in K-corrections
        if isinstance(good, tuple) and len(good) == 2 and isinstance(good[0], np.ndarray) and isinstance(good[1], np.ndarray):
            for i in range(ngood):
                    img_downscale[good[0][i]][good[1][i]] = r_maggies[i]/(1.0 + highz_info['redshift'])

        #convert image back to cts

        img_downscale = maggies2cts(img_downscale,highz_info['exptime'],highz_info['zp'])
        bg = maggies2cts(bg,highz_info['exptime'],highz_info['zp'])

        return img_downscale,bg,ngood

    

def ferengi(images,background,lowz_info,highz_info,namesout,imerr=None,err0_mag=None,kc_obj=None,noflux=False,evo=None,noconv=False,kcorrect=False,extend=False,nonoise=False,check_psf_FWHM=False,border_clip=3):

    #image , background, lowz_info['psf'], highz_info['psf'], imerr listas con los los path de los archivos, deben tener el mismo numero de entradas
    
    n_bands = len(images)

    if n_bands == 1:
        apply_kcorrect = False
    else:
        apply_kcorrect = True

    Pl = []
    Ph = []
    sky = []
    image = []
    im_err = []

    if lowz_info['lambda'] is not None:
        lambda_lo = np.array(lowz_info['lambda'])
    if highz_info['lambda'] is not None:
        lambda_hi = np.array(highz_info['lambda'])

    for i in range(n_bands):
        image.append(pyfits.getdata(images[i]))
        Pl.append(pyfits.getdata(lowz_info['psf'][i]))
        sky.append(pyfits.getdata(background[i]))

    lengths = [len(images), len(background), len(lowz_info['psf'])]
    if not all(length == lengths[0] for length in lengths):
        print('All input lists must have the same number of entries')
        return -99, -99, -99
    
    shapes = [banda.shape for banda in image]
    if len(set(shapes)) != 1:
        print("Error: All images must have the same shape")
        return -99,-99, -99
    

    Ph.append(pyfits.getdata(highz_info['psf']))
    input = image 
    
    if imerr is None:
        for i in range(n_bands):
            im_err.append(1/np.sqrt(np.abs(image[i])))
    else:
        for i in range(n_bands):
            im_err.append(pyfits.getdata(imerr[i]))

    
    if apply_kcorrect==False: #k false
        img_nok = maggies2cts(cts2maggies(image[0],lowz_info['exptime'],lowz_info['zp'][0]),highz_info['exptime'],highz_info['zp'])#*1000.
        psf_lo = Pl[0]
        psf_hi = Ph[0]

    else:
        #select best matching PSF for output redshift
        dz = np.abs(lambda_hi / lambda_lo - 1)
        idx_bestfilt = np.argmin(dz)
        psf_lo = Pl[idx_bestfilt]
        psf_hi = Ph[0]

    if(check_psf_FWHM == True and check_PSF_FWHM(psf_lo,psf_hi,lowz_info['pixscale'],highz_info['pixscale'],lowz_info['redshift'],highz_info['redshift']) == False): 
        
        print("the FWHM of the redshifted PSF is broader than that of the high-z PSF. The simulation is not possible")
        return -99, -99, -99



    if apply_kcorrect==False: #k false
        #scale the image down
        img_downscale = ferengi_downscale(img_nok,lowz_info['redshift'],highz_info['redshift'],lowz_info['pixscale'],highz_info['pixscale'],evo=evo,nofluxscale=noflux)
    else:

        img_downscale,bg,n_pixk = kcorrect_maggies(image,im_err,lowz_info,highz_info,lambda_lo,lambda_hi,err0_mag,evo,noflux,kc_obj)


    med = scndi.median_filter(img_downscale, size=3)
    idx = np.where(~np.isfinite(img_downscale))
    
    #remove infinite pixels: replace with median (3x3)
    if idx[0].size > 0:
        img_downscale[idx] = med[idx]
    
    #replace 0-value pixels with median (3x3)
    idx = np.where(img_downscale == 0)
    if idx[0].size > 0:
        img_downscale[idx] = med[idx]

    #k true
    if apply_kcorrect==True:
        m, sig, nrej = resistent_mean(img_downscale,3)#m = media, sig = desviacion estandar, nrej = numero de rechazos
        sig = sig * np.sqrt(np.size(img_downscale) - 1 - nrej)
        idx = np.where((np.abs(img_downscale) > 10 * sig) & (img_downscale!=bg)) #indices de img_downscale que cumplen con la condicion
        if idx[0].size > 0:
            print('High sigma pixels detected')
            fit = robust_linefit(np.abs(bg[idx]), np.abs(img_downscale[idx]))
            delta = np.abs(img_downscale[idx]) - (fit[0] + fit[1] * np.abs(bg[idx]))
            
            m, sig, nrej = resistent_mean(delta, 3)
            sig *= np.sqrt(img_downscale.size - 1 - nrej)
            
            idx1 = np.where(delta / sig > 50)
            if idx1[0].size > 0:
                img_downscale[idx[0][idx1]] = med[idx[0][idx1]]
        
    

    #subtracting sky here
    img_downscale -= ring_sky(img_downscale, 50, 15, nw=True)

 
    if noconv==True:
        dump_results(img_downscale/highz_info['exptime'],psf_lo/np.sum(psf_lo),images,background,namesout,lowz_info,highz_info)
        return img_downscale/highz_info['exptime'],psf_lo/np.sum(psf_lo)
    
    #calculate the transformation PSF
    try:
    
        psf_low,psf_high,psf_t = ferengi_transformation_psf(psf_lo,psf_hi,lowz_info['redshift'],highz_info['redshift'],lowz_info['pixscale'],highz_info['pixscale'])
        
    except TypeError as err:
        print('Enlarging PSF failed! Skipping Galaxy.')
        return -99,-99, -99

    try:
        # Asegurarnos de que tanto psf_lo como psf_t tengan el mismo número de dimensiones
        if psf_lo.ndim > 2 or psf_t.ndim > 2:
            # Si alguna de las matrices tiene una dimensión extra de tamaño 1, la eliminamos
            psf_lo = np.squeeze(psf_lo)
            psf_t = np.squeeze(psf_t)

        recon_psf = ferengi_psf_centre(apcon.convolve_fft(psf_lo,psf_t))
    except ZeroDivisionError as err:
        print('Reconstrution PSF failed!')
        return -99,-99, -99
    

    
    #normalise reconstructed PSF
    recon_psf/=np.sum(recon_psf)

    #convolve the high redshift image with the transformation PSF
    if n_bands == 1:
        img_downscale = ferengi_convolve_plus_noise(img_downscale/highz_info['exptime'],psf_t,sky[0],highz_info['exptime'],nonoise=nonoise,border_clip=border_clip,extend=extend)
        n_pixk = 0
    else:
        img_downscale = ferengi_convolve_plus_noise(img_downscale/highz_info['exptime'],psf_t,sky[idx_bestfilt],highz_info['exptime'],nonoise=nonoise,border_clip=border_clip,extend=extend)
    if np.amax(img_downscale) == -99:
        print('Sky Image not big enough!')
        return -99,-99

    dump_results(img_downscale,recon_psf,images,background,namesout,lowz_info,highz_info)
    
    return img_downscale,recon_psf,n_pixk

            
