import numpy as np
import math
import scipy.io as sio
import sys
sys.path.append('')

def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    # Fast and numerically precise:
    variance = np.average((values-average)**2, weights=weights)
    return (average, math.sqrt(variance))


def multivariate_gaussian(pos, mu, Sigma):

    """Return the multivariate Gaussian distribution on array pos.

 

    pos is an array constructed by packing the meshed arrays of variables

    x_1, x_2, x_3, ..., x_k into its _last_ dimension.

 
    """
 
    #Sigma is np.array([[ std_av, 1], [1, std_av]])
    
    n = mu.shape[0]
    Sigma_det = np.linalg.det(Sigma) # Compute the determinant of an array.
    Sigma_inv = np.linalg.inv(Sigma) # Compute the (multiplicative) inverse of a matrix.
    N = np.sqrt(((2*np.pi)**n) * Sigma_det)

    # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized

    # way across all the input variables.

    fac = np.einsum('...k,kl,...l->...', pos-mu, Sigma_inv, pos-mu)
    return np.exp(-fac / 2) / N


def extract_means_std():
    
    '''
    
    This funtion calculates the means
    vor the vertical data, horizontal
    data and their averaged standard deviation.
    
    These means were also used to make the RF maps
    for each electrodes.
    
    Makes use of modle.weighted_avg_and_std()
    
    To make the gaus map. 
    '''
    squeezed = sio.loadmat('RFfit.mat', squeeze_me=True)
    hori_means = []
    verti_means = []
    std_avg = []

    for electrodes in range(192):
        ELEC_DATA = squeezed['Elec']
        FIT = 2 
        HORIZONTAL = 0
        VERTICAL = 1
        SIGNAL = 0
        VIS_ANGLE = 1

        horizontal_cor = ELEC_DATA[electrodes][0][HORIZONTAL][FIT]
        vertical_cor = ELEC_DATA[electrodes][0][VERTICAL][FIT]

        hori_signal = horizontal_cor.flatten()[0][SIGNAL]
        hori_visual_angle = horizontal_cor.flatten()[0][VIS_ANGLE]

        verti_signal = vertical_cor.flatten()[0][SIGNAL]
        verti_visual_angle = vertical_cor.flatten()[0][VIS_ANGLE]

        prob_hori = hori_signal - hori_signal.min()
        prob_verti = verti_signal - verti_signal.min()

        hori_mean, hori_std = weighted_avg_and_std(hori_visual_angle, prob_hori)
        verti_mean, verti_std = weighted_avg_and_std(verti_visual_angle, prob_verti)

        std_av = (hori_std + verti_std)/2

        hori_means.append(hori_mean)
        verti_means.append(verti_mean)
        std_avg.append(std_av)
        
    return hori_means, verti_means, std_avg
    
def make_gausmaps(hori_means, verti_means, std_av, size):
    
    '''
    
    This function was used to make the RF maps based
    on the means and stds returned from the extract_means_std()
    function.
    
    '''
    
    N = 2.5 *size
    X = np.linspace(-7.5, 12.5, N)
    Y = np.linspace(-12.5, 7.5, N)
    X, Y = np.meshgrid(X, Y)

    # Mean vector and covariance matrix
    mu = np.array([hori_mean, verti_mean])
    Sigma = np.array([[ std_av, 0], [0, std_av]]) # the zeros are the covariance matrix (correlation between x and y)
    #In our case it is not needded so et to 0

    # Pack X and Y into a single 3-dimensional array
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y

    # The distribution on the variables X, Y packed into pos.
    Z = module.multivariate_gaussian(pos, mu, Sigma)
    Z = (Z/Z.sum())
    
    img_width = Z.shape[0]
    left_e = int((img_width / 100 * 27.5))
    right_s = int(img_width - (img_width/100*32.5))

    img_height = Z.shape[1]
    top_e = int((img_height / 100 * 27.5))
    bot_s = int(img_height - (img_height/100*32.5))
    
    Z_cropped = Z[left_e:right_s, top_e:bot_s]
    
    Z = Z_cropped
    gaussians_list.append(Z)
    
    return gaussian_list



#     return confidence_mask.any(0)

def make_confidence_mask(hori_means, verti_means, std_avg, size):
    std = std_avg
    hori_means_v = np.concatenate([hori_means[:144], hori_means[145:]])
    verti_means_v = np.concatenate([verti_means[:144], verti_means[145:]])
    std_v = np.concatenate([std[:144], std[145:]])

    hori_means_pixels = (hori_means_v + 2) * size/8
    verti_means_pixels = (verti_means_v+7) * size/8
    std_pixels = std_v*size/8

    confidence_radius = std_pixels
    meshgrid = np.mgrid[:size, :size]
    meshgrid = meshgrid[np.newaxis,:,:,:]
    meshgrid = meshgrid.repeat(191,axis=0)

    means_expanded_to_meshgrid = np.stack([verti_means_pixels,hori_means_pixels],axis=1)
    means_expanded_to_meshgrid = means_expanded_to_meshgrid[:,:,np.newaxis,np.newaxis]
    means_expanded_to_meshgrid = means_expanded_to_meshgrid.repeat(size, 2).repeat(size, 3)

    confidence_mask = (
        np.sqrt(((meshgrid - means_expanded_to_meshgrid)**2).sum(1)) 
           < confidence_radius[:,np.newaxis, np.newaxis].repeat(size, 1).repeat(size, 2)
    )      

    return confidence_mask.any(0)


# def make_confidence_mask(hori_means, verti_means, std_avg, size):
#     std = std_avg
#     hori_means_v = np.concatenate([hori_means[:144], hori_means[145:]])
#     verti_means_v = np.concatenate([verti_means[:144], verti_means[145:]])
#     std_v = np.concatenate([std[:144], std[145:]])

#     hori_means_pixels = (hori_means_v + 2) * 30
#     verti_means_pixels = (verti_means_v+7) * 30
#     std_pixels = std_v*30

#     confidence_radius = std_pixels
#     meshgrid = np.mgrid[:size, :size]
#     meshgrid = meshgrid[np.newaxis,:,:,:]
#     meshgrid = meshgrid.repeat(191,axis=0)

#     means_expanded_to_meshgrid = np.stack([verti_means_pixels,hori_means_pixels],axis=1)
#     means_expanded_to_meshgrid = means_expanded_to_meshgrid[:,:,np.newaxis,np.newaxis]
#     means_expanded_to_meshgrid = means_expanded_to_meshgrid.repeat(240, 2).repeat(240, 3)

#     confidence_mask = (
#         np.sqrt(((meshgrid - means_expanded_to_meshgrid)**2).sum(1)) 
#            < confidence_radius[:,np.newaxis, np.newaxis].repeat(240, 1).repeat(240, 2)
#     )