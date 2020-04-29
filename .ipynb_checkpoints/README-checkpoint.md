# RF_GANsynth

RF_GAN model has the same objective as the RFmasked, however makes use of an adversarial loss.

The adversarial loss contains a generator and discriminator (netG and netD). 

1. The `netD` is trained on the real images (target img) based on the loss between the output and the real label. (The loss is called `errD_real`)

2. Then the netG model takes in the real input (RFs) and gives an output (outputG1).

3. This outputG1 that came out of the generator is passed through the discriminator, and netD learns that this is the "fake" img. 

4. A second loss is calculated between this netD output and the fake_label. (errD_fake)  

5. The errD_real and errD_fake are added together to become errD, and backpropagation is performed on errD.

6. The Generator takes the input again, and gives some output (outputG2).

7. This outputG2 that came out of the generator is passed through the discriminator, and netD learns this time that this is the "real" img. 

8. The loss between the output of netD and real_label is called the `errG`.

9. The loss between the output of netG and target img is called `vgg_loss`.

10. `errG` and `vgg_loss` are summed and then the loss is used for optimization. 


In summary, there are two main parts in the adversarial loss. 

In the first part, the discriminator is trained on real and fake images. This is done by allowing the discriminator to see the target and labeling them real and the fake images and labeling them fake. Backpropagation is performed on the netD such that the weights changes to minimize both of these losses until the model can distinguish between the two.

The second part contains the training of the generator. This is done by changing the generator model that will ensure low loss between its output's label (int) and the real label (errG) AND low loss between the image and its target (vgg_loss).  



withoutNoise: Original amount of channels (191 fior selectrode)

withNoise: With an extra channel containing noise images.
