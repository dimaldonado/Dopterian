import kcorrect
import numpy as np
import scipy.ndimage as scndi
from astropy.cosmology import FlatLambdaCDM
from dopterian import cosmology as cosmos
from dopterian import helperFunctions

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
            img_downscale.append(helperFunctions.ferengi_downscale(image[i],lowz_info['redshift'],highz_info['redshift'],lowz_info['pixscale'],highz_info['pixscale'],evo=evo,nofluxscale=noflux))
            
            img_downscale[i]-=helperFunctions.ring_sky(img_downscale[i], 50,15,nw=True)
        
            imerr_downscale.append(helperFunctions.ferengi_downscale(im_err[i],lowz_info['redshift'],highz_info['redshift'],lowz_info['pixscale'],highz_info['pixscale'],evo=evo,nofluxscale=noflux))

            #convert the error from cts to mags
            imerr_downscale[i] = 2.5 / np.log(10) * imerr_downscale[i] / img_downscale[i]

            #calculate the flux in each pixel (convert image from cts to maggies)
            img_downscale[i] = helperFunctions.cts2maggies(img_downscale[i],lowz_info['exptime'][i],lowz_info['zp'][i])
            
        
        
        
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
            m, s, n = helperFunctions.resistent_mean(img_downscale[i], 3)
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

        img_downscale = helperFunctions.maggies2cts(img_downscale,highz_info['exptime'],highz_info['zp'])
        bg = helperFunctions.maggies2cts(bg,highz_info['exptime'],highz_info['zp'])

        return img_downscale,bg,ngood


    

    