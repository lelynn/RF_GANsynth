import torch
import torch.nn as nn
import torch.nn.functional as F

# ********************************************************************
#   D E C O N V O L U T I O N   AND   R E S I D U A L   M O D E L
# ********************************************************************
class ResblocksDeconv(nn.Module):
    def __init__(self, in_channels, outsize):
        super(ResblocksDeconv, self).__init__()
#         conv 1_1:
        self.conv1_1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1)
        self.bn1_1 = nn.BatchNorm2d(64)
#         conv 1_2
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn1_2 = nn.BatchNorm2d(64)
#         pool 1
        self.maxp1 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
#         120 x 120
#         conv2_1
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2_1 = nn.BatchNorm2d(128)
#         conv2_2
        self.dropout2 = nn.Dropout(0.5)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn2_2 = nn.BatchNorm2d(128)
#         pool2
        self.maxp2 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

#         60 x 60
#         conv3_1
        self.dropout1 = nn.Dropout(0.5)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn3_1 = nn.BatchNorm2d(256)
#         conv3_2
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn3_2 = nn.BatchNorm2d(256)
#         conv3_3
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn3_3 = nn.BatchNorm2d(256)
#         pool3
        self.maxp3 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)   
    
    
#         30 x 30
        self.dropout3 = nn.Dropout(0.5)
#         conv4_1
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn4_1 = nn.BatchNorm2d(512)
#         conv4_2
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn4_2 = nn.BatchNorm2d(512)
#         conv4_3
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn4_3 = nn.BatchNorm2d(512)
#         pool4
        self.maxp4 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)   
    
#         15 x 15 
#         conv5_1
        self.dropout4 = nn.Dropout(0.5)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn5_1 = nn.BatchNorm2d(512)
#         conv5_2
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn5_2 = nn.BatchNorm2d(512)
#         conv5_3
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn5_3 = nn.BatchNorm2d(512)
#         pool5
        self.maxp5 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)           
     
#         7 x 7 
        self.residualBlock_1 = ResidualBlock(512, 512)
        self.residualBlock_2 = ResidualBlock(512, 512)
        self.residualBlock_3 = ResidualBlock(512, 512)
        self.residualBlock_4 = ResidualBlock(512, 512)
        self.residualBlock_5 = ResidualBlock(512, 512)
        
        
#         7 x 7 
#         unpool5
        self.unpool5 = nn.MaxUnpool2d(2,2) 

#         _ x _
#         deconv5_1
        self.dropout5 = nn.Dropout(0.5)
        self.deconv5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.debn5_1 = nn.BatchNorm2d(512)
#         conv3_2
        self.deconv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.debn5_2 = nn.BatchNorm2d(512)
#         conv3_3
        self.deconv5_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.debn5_3 = nn.BatchNorm2d(512)
        
        
#         _ x _ 
#         unpool4
        self.unpool4 = nn.MaxUnpool2d(2,2) #paper did unpoolsize 28 (we want 30 x 30)
#         _ x _
#         deconv4_1
        self.dropout6 = nn.Dropout(0.5)
        self.deconv4_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.debn4_1 = nn.BatchNorm2d(512)
#         deconv4_2
        self.deconv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.debn4_2 = nn.BatchNorm2d(512)
#         conv4_3
        self.deconv4_3 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.debn4_3 = nn.BatchNorm2d(256)        
        
#         _ x _ 
#         unpool3
        self.unpool3 = nn.MaxUnpool2d(2,2) #paper did unpoolsize 56 (we want 60 x 60)
#         _ x _
#         deconv3_1
        self.dropout7 = nn.Dropout(0.5)
        self.deconv3_1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.debn3_1 = nn.BatchNorm2d(256)
#         deconv3_2
        self.deconv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.debn3_2 = nn.BatchNorm2d(256)
#         deconv3_3
        self.deconv3_3 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.debn3_3 = nn.BatchNorm2d(128)
        
        
#         _ x _ 
#         unpool2
        self.unpool2 = nn.MaxUnpool2d(2,2) #paper did unpoolsize 112 (we want 120 x 120)
#         _ x _
#         deconv3_1
        self.dropout8 = nn.Dropout(0.5)
        self.deconv2_1 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.debn2_1 = nn.BatchNorm2d(128)
#         deconv3_2
        self.deconv2_2 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.debn2_2 = nn.BatchNorm2d(64)
        
        
#         _ x _ 
#         unpool1
        self.unpool1 = nn.MaxUnpool2d(2,2) #paper did unpoolsize 224 (we want 240 x 240)
#         _ x _
#         deconv3_1
        self.deconv1_1 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.debn1_1 = nn.BatchNorm2d(32)
#         deconv3_2
        self.deconv1_2 = nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1)
#         self.debn1_2 = nn.BatchNorm2d(3)
        

        self.in_channels = in_channels
        self.outsize = outsize

    def __call__(self, x):
        h = self.conv1_1(x)
        h = self.bn1_1(h)
        h = F.relu(h)
        h = self.conv1_2(h)
        h = self.bn1_2(h)
        h = F.relu(h)
        h, i1 = self.maxp1(h)
        size1 = h.size()
        
        h = self.dropout1(h)
        h = self.conv2_1(h)
        h = self.bn2_1(h)
        h = F.relu(h)
        h = self.conv2_2(h)
        h = self.bn2_2(h)
        h = F.relu(h)
        h, i2 = self.maxp2(h)
        size2 = h.size()
        
        h = self.dropout2(h)
        h = self.conv3_1(h)
        h = self.bn3_1(h)
        h = F.relu(h)
        h = self.conv3_2(h)
        h = self.bn3_2(h)
        h = F.relu(h)
        h = self.conv3_3(h)
        h = self.bn3_3(h)
        h = F.relu(h)
        h, i3 = self.maxp3(h)
        size3 = h.size()
        
        h = self.dropout3(h)
        h = self.conv4_1(h)
        h = self.bn4_1(h)
        h = F.relu(h)
        h = self.conv4_2(h)
        h = self.bn4_2(h)
        h = F.relu(h)
        h = self.conv4_3(h)
        h = self.bn4_3(h)
        h = F.relu(h)
        h, i4 = self.maxp4(h)
        size4 = h.size()        
        
        h = self.dropout4(h)
        h = self.conv5_1(h)
        h = self.bn5_1(h)
        h = F.relu(h)
        h = self.conv5_2(h)
        h = self.bn5_2(h)
        h = F.relu(h)
        h = self.conv5_3(h)
        h = self.bn5_3(h)
        h = F.relu(h)
        h, i5 = self.maxp5(h)
        size5 = h.size()        

        h = self.residualBlock_1(h)
        h = self.residualBlock_2(h)
        h = self.residualBlock_3(h)
        h = self.residualBlock_4(h)
        h = self.residualBlock_5(h)


        h = self.unpool5(h, i5, output_size = size4)
        h = self.dropout5(h)
        h = self.deconv5_1(h)
        h = self.debn5_1(h)
        h = F.relu(h)
        h = self.deconv5_2(h)
        h = self.debn5_2(h)
        h = F.relu(h)
        h = self.deconv5_3(h)
        h = self.debn5_3(h)
        h = F.relu(h)
        
        h = self.unpool4(h, i4)
        h = self.dropout6(h)
        h = self.deconv4_1(h)
        h = self.debn4_1(h)
        h = F.relu(h)
        h = self.deconv4_2(h)
        h = self.debn4_2(h)
        h = F.relu(h)
        h = self.deconv4_3(h)
        h = self.debn4_3(h)
        h = F.relu(h)

        h = self.unpool3(h, i3)
        h = self.dropout7(h)
        h = self.deconv3_1(h)
        h = self.debn3_1(h)
        h = F.relu(h)
        h = self.deconv3_2(h)
        h = self.debn3_2(h)
        h = F.relu(h)
        h = self.deconv3_3(h)
        h = self.debn3_3(h)
        h = F.relu(h)

        
        h = self.unpool2(h, i2)
        h = self.dropout8(h)
        h = self.deconv2_1(h)
        h = self.debn2_1(h)
        h = F.relu(h)
        h = self.deconv2_2(h)
        h = self.debn2_2(h)
        h = F.relu(h)
        
        
        h = self.unpool1(h, i1)
        h = self.deconv1_1(h)
        h = self.debn1_1(h)
        h = F.relu(h)
        h = self.deconv1_2(h)
#         h = self.debn1_2(h)
#         h = F.relu(h)

        y = (torch.tanh(h) + 1) / 2

        return y
    
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.bn1= nn.BatchNorm2d(out_channels)
        self.bn2= nn.BatchNorm2d(out_channels)
        self.conv1 =nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1, bias = False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1, bias = False)

        self.in_channels = in_channels
        self.out_channels = out_channels

    def __call__(self, x, finetune = False):
        h = self.conv1(x)
        h = self.bn1(h)
        h = F.relu(h)
        
        h = self.conv2(h)
        h = self.bn2(h)
        y = F.relu(h + x)

        return y


