import kcorrect
import numpy as np

def kcorrect_image(data_filter_list, filter_list,redshift,imerr):

    if len(data_filter_list) != len(filter_list) or len(data_filter_list) != len(imerr):
        raise ValueError("El número de matrices en data_filter_lis tdebe ser igual al numero de matrices en imerr e igual de strings en filter_list.")

    if not all(((matrix.shape == data_filter_list[0].shape) and (matrix.shape==imerr[0].shape)) for matrix in data_filter_list):
        raise ValueError("Todas las matrices en data_filter_list deben ser del mismo tamaño y el tamaño de imerr debe ser igual al de las matrices en data_filter_list.")
    
    kc = kcorrect.kcorrect.Kcorrect(responses=filter_list)



    nbands = len(data_filter_list)
    shape_bands = data_filter_list[0].shape #tamaño de las matrices (bandas)

    print(kc.templates.wave)

    