import numpy as np
from tqdm import tqdm
import module_split as module


def make_unique_synth(set_t, images_set):
    '''Generates synthetic data for 191 channels using the seen images from the training map.
    
    set_t: Input the parameter set_t for indicating training or testing.
    
    images_set: Can be either 'synthetic' or 'CIFAR'.
    
    Saves an np.array of the following shape: (n_images, n_electrodes, width, height)
    where n_images is based on the unique images in the test or training set (test: 889, train: 7998)
    n_electrodes is 191, and img witdh x height = 240 x 240. 
    
    If you want to separate by layers (V1 & V4 respectively), then  
     
    
    v1 = inputs191_list[:, :97].sum(1)
    v4 = weighted_gaus[:, 97:].sum(1)

    inputs_2 = np.stack((v1,v4), axis=0)
    
    '''
    
    gaus = module.load_gausdata()
    nn_seen = module.load_ydata(images_set)
    
    data_indices = np.load(f'{set_t}/index_{set_t}_LFP_split.npy').astype(np.int)
    data_indices_unique = np.unique(data_indices)

    if set_t == 'training':
        slices = [slice(x*1000, (x+1)*1000) for x in range(8)]
    else:
        slices = [slice(None, None)]

    for batch_i, batch_slice in enumerate(slices):
        inputs191_list = []
        for data_index in tqdm(data_indices_unique[batch_slice]):
        # data_index = data_indices[0]

            synth_signal = nn_seen[data_index][:,:,:, np.newaxis].repeat(191, axis=3)
#             gaus_expand_to_batch = gaus.expand([191, 240, 240])
            synth_signal_sum4 = synth_signal.transpose(3,0,1,2).sum(3) # merge color dimensions

            dot_number = (gaus * synth_signal_sum4).sum((1,2))
#             weighted_gaus = gaus_expand_to_batch * dot_number[:, np.newaxis, np.newaxis].repeat((1, 240,240))

            inputs191_list.append(dot_number)

        inputs191_list = np.stack(inputs191_list)

        np.save(f'{set_t}/{set_t}_{images_set[:5]}191batch{batch_i}', inputs191_list)

        
