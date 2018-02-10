from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from Architecture.dcgan import Generator
from Architecture.dcgan import Discriminator
from Architecture.dcgan import weights_init as weights_init
from Cepa.aug import Data_Augmentation

print(torch.__version__)

batchSize = 64 
imageSize = 64 

#----------STEP 1---------------------
#       LOADING DATASET
#-------------------------------------
d_aug_obj = Data_Augmentation(imageSize,'./dbz_dataset')
augmented_dataset = d_aug_obj.data_augmentation()

print('Dataset length : {}'.format(len(augmented_dataset)))

dataloader = torch.utils.data.DataLoader(augmented_dataset, batch_size = batchSize, shuffle = True, num_workers = 2) # We use dataLoader to get the images of the training set batch by batch.

#-----------STEP 2a)------------------
#CREATING & INITIALIZING GENERATOR
#-------------------------------------
netG = Generator() 
netG.apply(weights_init) 

#----------STEP 2b)-------------------
#CREATING & INITIALIZING DISCRIMINATOR
#-------------------------------------
netD = Discriminator() 
netD.apply(weights_init)
 

#----------STEP 3)--------------------
#     CHOOSE LOSS & OPTIMIZATORS
#-------------------------------------
criterion = nn.BCELoss()
optimizerD = optim.Adam(netD.parameters(), lr = 0.0002, betas = (0.5,0.999))#sve ove vrednosti su proizisle iz eksperimenata
optimizerG = optim.Adam(netG.parameters(), lr = 0.0002, betas = (0.5,0.999))#sve ove vrednosti su proizisle iz eksperimenata

#----------STEP 4)--------------------
#       TRAINING DCGAN
#-------------------------------------

for epoch in range(40):
    for i, data in enumerate(dataloader, 0):
        
        #----------STEP 4 a)-------------------
        #  TRAINING DISCRIMINATOR ON BATCH OF
        #             REAL DATA
        #--------------------------------------
        
        netD.zero_grad()
        real, _ = data 
        input = Variable(real)
        target  = Variable(torch.ones(input.size()[0])) 
        output = netD(input) 
        errD_real = criterion(output,target) 
        
        #----------STEP 4 b)-------------------
        #  TRAINING DISCRIMINATOR ON BATCH OF
        #             FAKE DATA
        #--------------------------------------
        
        
        noise = Variable(torch.randn(input.size()[0], 100,1,1))
        fake = netG(noise) 
        target = Variable(torch.zeros(input.size()[0])) 
        output = netD(fake.detach())
        errD_fake = criterion(output, target)
        
        #----------STEP 4 c)-------------------
        #CALCULATING TOTAL ERROR & BACKPROPAGING
        #  THROUGH ENTIRE NETWORK ARCHITECTURE
        #--------------------------------------
        errD = errD_real + errD_fake #total error
        
        errD.backward() 
        #----------STEP 4 d)-------------------
        #     UPDATING PARAMETERES (WEIGHTS)
        #--------------------------------------
        optimizerD.step()
        
        #----------STEP 5 a)-------------------
        #    TRAINING GENERATOR TO MAXIMIZE
        # LIKELIHOOD OF DISCRIMINATOR BEING WRONG
        #--------------------------------------
        netG.zero_grad() 
        target  = Variable(torch.ones(input.size()[0])) 
        output = netD(fake) 
        errG = criterion(output, target)
           
        #----------STEP 4 c)-------------------
        #CALCULATING TOTAL ERROR & BACKPROPAGING
        #  THROUGH ENTIRE NETWORK ARCHITECTURE
        #--------------------------------------
        errG.backward(retain_graph=True)
        
        #----------STEP 4 d)-------------------
        #     UPDATING PARAMETERES (WEIGHTS)
        #--------------------------------------
       
        optimizerG.step()
          
        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f' %(epoch,40,i,len(dataloader),errD.data[0],errG.data[0]))
        if i % 100 == 0:
            vutils.save_image(real, '%s/real_samples.png' %"./results",normalize= True)
            fake = netG(noise)
            vutils.save_image(fake.data, '%s/fake_samples_epoch_%03d.png' % ("./results",epoch),normalize= True)
