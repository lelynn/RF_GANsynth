
import torch.nn as nn
import random
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import torch.utils.data as dataset
from tqdm import tqdm

import model_file
import module_Noise as module

import RF_module as RF

import torchvision.transforms as transforms

import os

# =================================================
# D O U B L E   C H E C K   B E F O R E   R U N :  
# =================================================
load_epoch = True
epoch_loaded = 966
old_runname = 'RFSynthGAN_Noise_gl_0.5'
runname = 'RFSynthGAN_Noise_gl_0.05'
images_set = 'synthetic'
# =================================================


manualSeed = 999
random.seed(manualSeed)
torch.manual_seed(manualSeed)

device = 1
cuda0 = torch.device(f'cuda:{device}')
batch_size = 32
all_image_size = 96
num_epochs = 1000

lr = 0.0002

# Beta1 hyperparam for Adam optimizers
beta1 = 0.8
vgg_beta = 1
G_beta = 0.05
# -----
# Models
# -----
in_channels=192
netG = model_file.ResblocksDeconv(in_channels, (all_image_size,all_image_size))
if load_epoch: 
    netG.load_state_dict(torch.load(f'{old_runname}/netG_epochs_{epoch_loaded}.model'))
else: 
    netG.apply(module.weights_init)

netD = module.Discriminator().to(device)
if load_epoch: 
    netD.load_state_dict(torch.load(f'{old_runname}/netD_epochs_{epoch_loaded}.model', map_location='cpu'))
else:
    netD.apply(module.weights_init)

    
if __name__ == '__main__':    
    if device >= 0:
        netG.cuda(device)
        netD.cuda(device)


    lossFunction = nn.BCELoss()
    vgg_lossFunction = module.VGGLoss(device)

    if in_channels == 3:
        inputtype = 'V1_V4'
    if in_channels == 192:
        inputtype = 'all_channels'
        

    # -----
    # RF gaus maps
    # ------
    gaus = module.load_gausdata(size= '96')

    seen_images = module.load_ydata(None, size='96')
    seen_images_torch = torch.from_numpy(seen_images)

    # ------
    # Training
    # ------
    dot_numbers_train = np.load(f'training/training_{images_set[:5]}191final.npy')
    training_iterator = module.make_iterator_unique(dot_numbers_train, 'training', batch_size, shuffle = True)

    # ------
    # Testing
    # ------
    dot_numbers_test = np.load(f'testing/testing_{images_set[:5]}191final.npy')
    testing_iterator = module.make_iterator_unique(dot_numbers_test, 'testing', batch_size, shuffle = False)


    real_label = 1
    fake_label = 0

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    hori_means, verti_means, std_avg = RF.extract_means_std()


    confidence_mask = RF.make_confidence_mask(hori_means, verti_means, std_avg, size = 96)
    confidence_mask = torch.from_numpy(confidence_mask.astype('float32')).to(cuda0)


    # Losses to append for TRAINING:
    G_vgg_losses_train=[]
    G_losses_train=[]
    vgg_losses_train=[]

    D_losses_train=[]
    D_losses_real_train=[]
    D_losses_fake_train=[]

    # Losses to append for TESTING:
    G_vgg_losses_test=[]
    G_losses_test=[]
    vgg_losses_test=[]

    D_losses_test=[]
    D_losses_real_test=[]
    D_losses_fake_test=[]

    iters = 0

    for epoch in range(num_epochs):
        netG.train(True)
        G_vgg_loss_train = 0
        G_loss_train = 0
        vgg_loss_train = 0

        D_loss_train = 0
        D_loss_real_train = 0
        D_loss_fake_train = 0

        for dot_number, img_indices in tqdm(training_iterator, total=len(training_iterator)):

            # -----
            # Inputs
            # -----
            gaus_expand_to_batch = gaus.expand([len(img_indices), 191, all_image_size, all_image_size])
            weight_images = dot_number[:,:,np.newaxis, np.newaxis].expand([len(img_indices), 191, all_image_size, all_image_size])     
            fixed_noise = torch.randn(len(img_indices), 1, 96, 96)
            # We want to use the dot number and repeat it (expand to gaus) such that it will have the same shape. 
            #Then you multiply with the gaus_exapnd_go_batch!
            inputs = module.select_type_inputs(inputtype, gaus_expand_to_batch, weight_images, fixed_noise)
            inputs = inputs.to(cuda0)


            # -----
            # Targets
            # -----
            target_batch = seen_images_torch[img_indices]
            target_batch = target_batch.transpose(3,1).transpose(2,3)
            target_batch = target_batch.to(cuda0)
            target_batch *= confidence_mask.expand_as(target_batch)

            # ==================================================================
            # D I S C R I M I N A T O R| Maximizing log(D(x)) + log(1 - D(G(z)))
            # ==================================================================

            netD.zero_grad()
            netG.zero_grad()
            # -------------------------
            # Train discr. on REAL img
            # -------------------------
            b_size = target_batch.size(0)

            label = torch.full((b_size,), real_label, device=device)
            label.fill_(real_label)  # fake labels are real for generator cost
            outputDreal = netD(target_batch).view(-1)

            errD_real = lossFunction(outputDreal, label)        

            # -------------------------
            # Train discr. on FAKE img
            # -------------------------
            outputGfake = netG(inputs) 
            outputGfake *= confidence_mask.expand_as(outputGfake)

            label = torch.full((b_size,), fake_label, device=device)
            label.fill_(fake_label)
            outputDfake = netD(outputGfake.detach()).view(-1)

            errD_fake = lossFunction(outputDfake, label)

            # Add the gradients from the all-real and all-fake batches
            errD = errD_real + errD_fake
            errD.backward()
            optimizerD.step()



            # ==================================================================
            # G E N E R A T O R| maximize log(D(G(z)))
            # ==================================================================
            # ------------------------------------------------------------------
            # Train generator to fool the discriminator and learn target images
            # ------------------------------------------------------------------
            netG.zero_grad()
            label = torch.full((b_size,), real_label, device=device)
            label.fill_(real_label)
            # ------------------------------------------------------------------
            # a forward pass through the generator
            # ------------------------------------------------------------------
            outputGfake = netG(inputs) 
            outputGfake *= confidence_mask.expand_as(outputGfake)
            outputDfake = netD(outputGfake).view(-1)
            # ------------------------------------------------------------------
            # Fake images (determined by distriminator) should become more real
            # ------------------------------------------------------------------
            errG = lossFunction(outputDfake, label)
            # ------------------------------------------------------------------
            # WHILE using vgg loss to generate closer to target.
            # ------------------------------------------------------------------
            vgg_loss = vgg_lossFunction(outputGfake, target_batch) 
            # ------------------------------------------------------------------
            # Combine both losses: with a beta value for vgg
            # ------------------------------------------------------------------
            errG_vgg = (errG* G_beta) + (vgg_loss * vgg_beta)

            errG_vgg.backward()                
            optimizerG.step()

            G_vgg_loss_train += errG_vgg.sum().item()
            G_loss_train += errG.sum().item()
            vgg_loss_train += vgg_loss.sum().item()

            D_loss_train += errD.sum().item()
            D_loss_real_train += errD_real.sum().item()
            D_loss_fake_train += errD_fake.sum().item()             

        G_vgg_losses_train.append(G_vgg_loss_train/len(training_iterator.sampler))
        G_losses_train.append(G_loss_train/len(training_iterator.sampler))
        vgg_losses_train.append(vgg_loss_train/len(training_iterator.sampler))

        D_losses_train.append(D_loss_train/len(training_iterator.sampler))
        D_losses_real_train.append(D_loss_real_train/len(training_iterator.sampler))
        D_losses_fake_train.append(D_loss_fake_train/len(training_iterator.sampler))

        # ------------------
        # TESTING
        # ------------------

        with torch.no_grad():
            netG.train(False)
            netG.eval()

            G_vgg_loss_test = 0
            G_loss_test = 0
            vgg_loss_test = 0

            D_loss_test = 0
            D_loss_real_test = 0
            D_loss_fake_test = 0

            for dot_number, img_indices in tqdm(testing_iterator, total=len(testing_iterator)):

                # -----
                # Inputs
                # -----
                gaus_expand_to_batch = gaus.expand([len(img_indices), 191, all_image_size, all_image_size])
                weight_images = dot_number[:,:,np.newaxis, np.newaxis].expand([len(img_indices), 191, all_image_size, all_image_size])     
                fixed_noise = torch.randn(len(img_indices), 1, 96, 96)
                # We want to use the dot number and repeat it (expand to gaus) such that it will have the same shape. 
                #Then you multiply with the gaus_exapnd_go_batch!
                inputs = module.select_type_inputs(inputtype, gaus_expand_to_batch, weight_images, fixed_noise)
                inputs = inputs.to(cuda0)

                # -----
                # Targets
                # -----
                target_batch = seen_images_torch[img_indices]
                target_batch = target_batch.transpose(3,1).transpose(2,3)
                target_batch = target_batch.to(cuda0)
                target_batch *= confidence_mask.expand_as(target_batch)

                # ==================================================================
                # D I S C R I M I N A T O R| testing
                # ==================================================================
                # -------------------------
                # TEST discr. on REAL img
                # -------------------------
                b_size = target_batch.size(0)

                label = torch.full((b_size,), real_label, device=device)
                label.fill_(real_label)  # fake labels are real for generator cost
                outputDreal = netD(target_batch).view(-1)

                errD_real = lossFunction(outputDreal, label)        

                # -------------------------
                # TEST discr. on FAKE img
                # -------------------------
                outputGfake = netG(inputs) 
                outputGfake *= confidence_mask.expand_as(outputGfake)

                label = torch.full((b_size,), fake_label, device=device)
                label.fill_(fake_label)
                outputDfake = netD(outputGfake.detach()).view(-1)

                errD_fake = lossFunction(outputDfake, label)

                # Add the gradients from the all-real and all-fake batches
                errD = errD_real + errD_fake


                # ==================================================================
                # G E N E R A T O R| testing
                # ==================================================================
                # ------------------------------------------------------------------
                # TESTING generator: does it fool discrim?
                # ------------------------------------------------------------------
                label = torch.full((b_size,), real_label, device=device)
                label.fill_(real_label)
                # ------------------------------------------------------------------
                # a forward pass through the generator
                # ------------------------------------------------------------------
                outputGfake = netG(inputs) 
                outputGfake *= confidence_mask.expand_as(outputGfake)
                outputDfake = netD(outputGfake).view(-1)
                # ------------------------------------------------------------------
                # Fake images (determined by distriminator) should become more real
                # ------------------------------------------------------------------
                errG = lossFunction(outputDfake, label)
                # ------------------------------------------------------------------
                # WHILE using vgg loss to generate closer to target.
                # ------------------------------------------------------------------
                vgg_loss = vgg_lossFunction(outputGfake, target_batch) 
                # ------------------------------------------------------------------
                # Combine both losses: with a beta value for vgg
                # ------------------------------------------------------------------
                errG_vgg = (errG * G_beta) + (vgg_loss * vgg_beta)

                G_vgg_loss_test += errG_vgg.sum().item()
                G_loss_test += errG.sum().item()
                vgg_loss_test += vgg_loss.sum().item()

                D_loss_test += errD.sum().item()
                D_loss_real_test += errD_real.sum().item()
                D_loss_fake_test += errD_fake.sum().item()

            G_vgg_losses_test.append(G_vgg_loss_test/len(testing_iterator.sampler))
            G_losses_test.append(G_loss_test/len(testing_iterator.sampler))
            vgg_losses_test.append(vgg_loss_test/len(testing_iterator.sampler))

            D_losses_test.append(D_loss_test/len(testing_iterator.sampler))
            D_losses_real_test.append(D_loss_real_test/len(testing_iterator.sampler))
            D_losses_fake_test.append(D_loss_fake_test/len(testing_iterator.sampler))
            
            
        # ===================
        # S A V I N G: losses
        # ===================
        # ------------------------------------------------------------------
        # TRAINING
        # ------------------------------------------------------------------
        os.makedirs(runname, exist_ok=True)
        
        np.save(f'{runname}/G_vgg_loss_train', np.array(G_vgg_losses_train))
        np.save(f'{runname}/Gloss_train', np.array(G_losses_train))
        np.save(f'{runname}/vggloss_train', np.array(vgg_losses_train))

        np.save(f'{runname}/Dloss_train', np.array(D_losses_train))
        np.save(f'{runname}/Dloss_real_train', np.array(D_losses_real_train))
        np.save(f'{runname}/Dloss_fake_train', np.array(D_losses_fake_train))
        # ------------------------------------------------------------------
        # TESTING
        # ------------------------------------------------------------------
        np.save(f'{runname}/G_vgg_loss_test', np.array(G_vgg_losses_test))
        np.save(f'{runname}/Gloss_test', np.array(G_losses_test))
        np.save(f'{runname}/vggloss_test', np.array(vgg_losses_test))

        np.save(f'{runname}/Dloss_test', np.array(D_losses_test))
        np.save(f'{runname}/Dloss_real_test', np.array(D_losses_real_test))
        np.save(f'{runname}/Dloss_fake_test', np.array(D_losses_fake_test))
        
        if load_epoch:
            torch.save(netG.state_dict(), f'{runname}/netG_epochs_{epoch+epoch_loaded+1}.model')
            torch.save(netD.state_dict(), f'{runname}/netD_epochs_{epoch + epoch_loaded + 1}.model')
            print('epochs: ', epoch+epoch_loaded+1)

        else:
            torch.save(netG.state_dict(), f'{runname}/netG_epochs_{epoch}.model')
            torch.save(netD.state_dict(), f'{runname}/netD_epochs_{epoch}.model')
            print('epochs: ', epoch)

    torch.save(netG.state_dict(), f'{runname}/netG_final.model')
    torch.save(netD.state_dict(), f'{runname}/netD_final.model')