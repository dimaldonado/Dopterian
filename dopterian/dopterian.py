import numpy as np
import scipy.ndimage as scndi
import astropy.io.fits as pyfits
import astropy.convolution as apcon
import matplotlib.pyplot as plt
from kcorrections import kcorrections
from mpl_toolkits.axes_grid1 import make_axes_locatable
from dopterian import helperFunctions


version = '1.0.0'   

  

    
def ferengi(imgname,background,lowz_info,highz_info,namesout,imerr=None,noflux=False,evo=None,noconv=False,kcorrect=False,extend=False,nonoise=False,border_clip=3):

    Pl=pyfits.getdata(lowz_info['psf'])
    Ph=pyfits.getdata(highz_info['psf'])
    sky=pyfits.getdata(background)
    image=pyfits.getdata(imgname)

    if imerr is None:
        imerr=1/np.sqrt(np.abs(image))
    else:
        imerr=pyfits.getdata(imerr)
    
    if kcorrect:
        #primero debemos aplicar ferengi downscale a todas las bandas de la entrada

        # luego ferengi downscale a imerr
        raise NotImplementedError('K-corrections are not implemented yet')
    else:
        img_nok = helperFunctions.maggies2cts(helperFunctions.cts2maggies(image,lowz_info['exptime'],lowz_info['zp']),highz_info['exptime'],highz_info['zp'])#*1000.
        img_downscale = helperFunctions.ferengi_downscale(img_nok,lowz_info['redshift'],highz_info['redshift'],lowz_info['pixscale'],highz_info['pixscale'],evo=evo,nofluxscale=noflux)

        psf_low = Pl
        psf_hi = Ph
    



    median = scndi.median_filter(img_downscale,3)
    idx = np.where(np.isfinite(img_downscale)==False)
    img_downscale[idx]=median[idx]
    
    idx = np.where(img_downscale==0.0)
    img_downscale[idx]=median[idx]
    
    X=img_downscale.shape[0]*0.5
    Y=img_downscale.shape[1]*0.5 ## To be improved
    
    img_downscale-=helperFunctions.ring_sky(img_downscale, 50,15,x=X,y=Y,nw=True)
    
    if noconv==True:
        helperFunctions.dump_results(img_downscale/highz_info['exptime'],psf_low/np.sum(psf_low),imgname,background,namesout,lowz_info,highz_info)
        return img_downscale/highz_info['exptime'],psf_low/np.sum(psf_low)

    try:
        psf_low,psf_high,psf_t = helperFunctions.ferengi_transformation_psf(psf_low,psf_hi,lowz_info['redshift'],highz_info['redshift'],lowz_info['pixscale'],highz_info['pixscale'])
        
    except TypeError as err:
        print('Enlarging PSF failed! Skipping Galaxy.')
        return -99,-99

    try:
        recon_psf = helperFunctions.ferengi_psf_centre(apcon.convolve_fft(psf_low,psf_t))
    except ZeroDivisionError as err:
        print('Reconstrution PSF failed!')
        return -99,-99
##    pyfits.writeto('transform_psf_dopterian.fits',psf_t,clobber=True)
    
    recon_psf/=np.sum(recon_psf)
    
    img_downscale = helperFunctions.ferengi_convolve_plus_noise(img_downscale/highz_info['exptime'],psf_t,sky,highz_info['exptime'],nonoise=nonoise,border_clip=border_clip,extend=extend)
    if np.amax(img_downscale) == -99:
        print('Sky Image not big enough!')
        return -99,-99
    


    helperFunctions.dump_results(img_downscale,recon_psf,imgname,background,namesout,lowz_info,highz_info)
    return img_downscale,recon_psf


def ferengi_k(images,background,lowz_info,highz_info,namesout,imerr=None,err0_mag=None,kc_obj=None,noflux=False,evo=None,noconv=False,kcorrect=False,extend=False,nonoise=False,border_clip=3):

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
        return -99, -99
    
    shapes = [banda.shape for banda in image]
    if len(set(shapes)) != 1:
        print("Error: All images must have the same shape")
        return -99,-99
    

    Ph.append(pyfits.getdata(highz_info['psf'][0]))
    input = image 
    
    if imerr is None:
        for i in range(n_bands):
            im_err.append(1/np.sqrt(np.abs(images[i])))
    else:
        for i in range(n_bands):
            im_err.append(pyfits.getdata(imerr[i]))

    
    if apply_kcorrect==False: #k false
        img_nok = helperFunctions.maggies2cts(helperFunctions.cts2maggies(image,lowz_info['exptime'],lowz_info['zp']),highz_info['exptime'],highz_info['zp'])#*1000.
        psf_lo = Pl[0]
        psf_hi = Ph[0]
    
    else:
        #select best matching PSF for output redshift
        dz = np.abs(lambda_hi / lambda_lo - 1)
        idx_bestfilt = np.argmin(dz)
        psf_lo = Pl[idx_bestfilt]
    
    if apply_kcorrect==False: #k false
        #scale the image down
        img_downscale = helperFunctions.ferengi_downscale(img_nok,lowz_info['redshift'],highz_info['redshift'],lowz_info['pixscale'],highz_info['pixscale'],evo=evo,nofluxscale=noflux)
    else:

        img_downscale,bg,n_pixk = kcorrections.kcorrect_maggies(image,im_err,lowz_info,highz_info,lambda_lo,lambda_hi,err0_mag,evo,noflux,kc_obj)


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
        m, sig, nrej = helperFunctions.resistent_mean(img_downscale,3)#m = media, sig = desviacion estandar, nrej = numero de rechazos
        sig = sig * np.sqrt(np.size(img_downscale) - 1 - nrej)
        idx = np.where((np.abs(img_downscale) > 10 * sig) & (img_downscale!=bg)) #indices de img_downscale que cumplen con la condicion
        if idx[0].size > 0:
            print('High sigma pixels detected')
            fit = helperFunctions.robust_linefit(np.abs(bg[idx]), np.abs(img_downscale[idx]))
            delta = np.abs(img_downscale[idx]) - (fit[0] + fit[1] * np.abs(bg[idx]))
            
            m, sig, nrej = helperFunctions.resistent_mean(delta, 3)
            sig *= np.sqrt(img_downscale.size - 1 - nrej)
            
            idx1 = np.where(delta / sig > 50)
            if idx1[0].size > 0:
                img_downscale[idx[0][idx1]] = med[idx[0][idx1]]
        
    

    #subtracting sky here
    img_downscale -= helperFunctions.ring_sky(img_downscale, 50, 15, nw=True)


    #graficar
    '''
    plt.figure()
    plt.imshow(img_downscale, cmap='gray')
    plt.title('Science Data Input')
    plt.colorbar()
    plt.show()
    '''
    if noconv==True:
        #dump_results(img_downscale/highz_info['exptime'],psf_lo/np.sum(psf_lo),images[filt_i],background[filt_i],namesout,lowz_info,highz_info)
        return img_downscale/highz_info['exptime'],psf_lo/np.sum(psf_lo)
    
    #calculate the transformation PSF
    try:
        psf_hi = Ph[0]
        psf_low,psf_high,psf_t = helperFunctions.ferengi_transformation_psf(psf_lo,psf_hi,lowz_info['redshift'],highz_info['redshift'],lowz_info['pixscale'],highz_info['pixscale'])
        
    except TypeError as err:
        print('Enlarging PSF failed! Skipping Galaxy.')
        return -99,-99

    try:
        recon_psf = helperFunctions.ferengi_psf_centre(apcon.convolve_fft(psf_lo,psf_t))
    except ZeroDivisionError as err:
        print('Reconstrution PSF failed!')
        return -99,-99
##    pyfits.writeto('transform_psf_dopterian.fits',psf_t,clobber=True)
    
    
    #normalise reconstructed PSF
    recon_psf/=np.sum(recon_psf)

    #convolve the high redshift image with the transformation PSF
    
    img_downscale = helperFunctions.ferengi_convolve_plus_noise(img_downscale/highz_info['exptime'],psf_t,sky[idx_bestfilt],highz_info['exptime'],nonoise=nonoise,border_clip=border_clip,extend=extend)
    if np.amax(img_downscale) == -99:
        print('Sky Image not big enough!')
        return -99,-99
    
    #graficar
    '''
    n_images = n_bands + 1  # +1 para incluir img_downscale

    # Crear la figura y los ejes
    fig, axes = plt.subplots(1, n_images, figsize=(15, 5))
    fig.suptitle("Comparación de Imágenes", fontsize=16)

    # Mostrar img_downscale en el primer subplot
    ax = axes[0]
    im = ax.imshow(img_downscale, origin='lower', cmap='gray')
    ax.set_title("Output")

    # Crear un eje para la colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')

    # Mostrar las imágenes de input en los subplots restantes
    for i, img in enumerate(input, start=1):
        ax = axes[i]
        im = ax.imshow(img, origin='lower', cmap='gray')
        ax.set_title("input "+lowz_info['filter'][i-1])
        
        # Crear un eje para la colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax, orientation='vertical')

    plt.tight_layout()
    plt.show()

    '''

    #dump_results(img_downscale,recon_psf,images[filt_i],background,namesout,lowz_info,highz_info)
    return img_downscale,recon_psf,n_pixk

            


if __name__=='__main__':
    PlowName='psf_sdss.fits'
    PhighName='psf_acs.fits'
    BgName='sky_ACSTILE_40x40.fits'
    InputImName='galaxy.fits'
    
    lowz_info = {'redshift':0.017,'psf':PlowName,'zp':28.235952,'exptime':53.907456,'filter':'r','lam_eff':6185.0,'pixscale':0.396}
    highz_info = {'redshift':0.06,'psf':PhighName,'zp':25.947,'exptime':6900.,'filter':'f814w','lam_eff':8140.0,'pixscale':0.03}
    
    import time as t
    t0=t.time()

#    imOUT,psfOUT = ferengi(InputImName,BgName,lowz_info,highz_info,['smooth_galpy_evo.fits','smooth_psfpy_evo.fits'],noconv=False,evo=lum_evolution)
    imOUT,psfOUT = ferengi(InputImName,BgName,lowz_info,highz_info,['smooth_galpy.fits','smooth_psfpy.fits'],noconv=False,evo=None)
    print('elapsed %.6f secs'%(t.time()-t0))

#    fig,ax=mpl.subplots(1,2)
#    ax[0].imshow(imOUT)
#    ax[1].imshow(psfOUT)
#    mpl.show()
