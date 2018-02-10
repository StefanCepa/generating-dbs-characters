import torch
import torch.nn as nn
import torch.nn.functional as F

   
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1: 
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__() 
      
        self.main = nn.Sequential(
                nn.ConvTranspose2d(100,512,4,1,0,bias = False),
                nn.BatchNorm2d(512),
                nn.ReLU(True),
                #nn.SELU(True),
                nn.ConvTranspose2d(512,256,4,2,1,bias = False),
                nn.BatchNorm2d(256),
                nn.ReLU(True),
                #nn.SELU(True),
                nn.ConvTranspose2d(256,128,4,2,1,bias = False),
                nn.BatchNorm2d(128),
                nn.ReLU(True),
                #nn.SELU(True),
                nn.ConvTranspose2d(128,64,4,2,1,bias = False),
                nn.BatchNorm2d(64),
                nn.ReLU(True),
                #nn.SELU(True),
                nn.ConvTranspose2d(64, 3,4,2,1,bias = False),
                nn.Tanh()
        ) 
        
    def forward(self, input): 
        output = self.main(input) 
        return output
    
class Discriminator(nn.Module):
    
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential( 
            nn.Conv2d(3, 64, 4, 2, 1, bias = False), 
            nn.LeakyReLU(0.2, inplace = True), 
            #nn.SELU(True),
            nn.Conv2d(64, 128, 4, 2, 1, bias = False), 
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace = True),
            #nn.SELU(True), 
            nn.Conv2d(128, 256, 4, 2, 1, bias = False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace = True),
            #nn.SELU(True),
            nn.Conv2d(256, 512, 4, 2, 1, bias = False), 
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace = True), 
            #nn.SELU(True),
            nn.Conv2d(512, 1, 4, 1, 0, bias = False), 
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.main(input)
        return output.view(-1) #Rezultat poslednjeg konvolucionog sloja je potrebno izravnati ( flaten )
    
 
        