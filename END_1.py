import torch
import torch.nn as nn
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self, x):
        return self.conv(x)
    
    
    

def double_conv01(in_channels, out_channels,f_size):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, f_size, padding='same'),
        nn.BatchNorm2d(num_features=out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, f_size, padding='same'),
        nn.BatchNorm2d(num_features=out_channels),
        nn.ReLU(inplace=True),
    )

def double_conv11(in_channels, out_channels,f_size,p_size):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, f_size,padding=p_size,stride=2),
        nn.BatchNorm2d(num_features=out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, f_size, padding='same'),
        nn.BatchNorm2d(num_features=out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, f_size, padding='same'),
        nn.BatchNorm2d(num_features=out_channels),
        nn.ReLU(inplace=True),
    ) 

def double_conv_u1(in_channels, out_channels,f_size):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, f_size, padding='same'),
        nn.BatchNorm2d(num_features=out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, f_size, padding='same'),
        nn.BatchNorm2d(num_features=out_channels),
        nn.ReLU(inplace=True),
    ) 

def trans_1(in_channels, out_channels,f_size,st_size):
    return nn.Sequential(
       nn.ConvTranspose2d(in_channels,out_channels, kernel_size=f_size, stride=st_size),
       nn.ReLU(inplace=True),
    ) 


class END_1(nn.Module):
    def __init__(self):
        super(END_1, self).__init__()

        self.inc = DoubleConv(3, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024 // 2)
        
        
        self.up1 = Up(1024, 512 // 2)
        self.up2 = Up(512, 256 // 2)
        self.up3 = Up(256, 128 // 2)
        self.up4 = Up(128, 64)
        self.outc = OutConv(64, 1)
        self.activation = torch.nn.Sigmoid()
        
        self.up_ = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        
        self.dconv_down1 = double_conv01(3, 64,(3,3))
        self.dconv_down2 = double_conv11(64, 128,(3,3),(1,1))
        self.dconv_down3 = double_conv11(128, 256,(3,3),(1,1))
        self.dconv_down4 = double_conv01(256, 512,(3,3))
        self.dconv_down5 = double_conv01(512, 512,(3,3))
        
        self.up11 = trans_1(256,256,  2,2)
        self.up22 = trans_1(128, 128, 2,2)
        
        
        self.m = nn.Dropout(p=0.10)

        
        self.dconv_up0 = double_conv_u1(512, 512,(3,3))
        self.dconv_up1 = double_conv_u1(512 + 512, 512,(3,3))
        self.dconv_up2 = double_conv_u1(512+256, 256,(3,3))
        self.dconv_up3 = double_conv_u1(256+128,128,(3,3))
        self.dconv_up4 = double_conv_u1(192,64,(3,3))
        self.conv_last = OutConv(64, 1)

    def forward(self, img,img_patches):
        ##encoder ##
        x1 = self.inc(img)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        ##decoder###
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x_steel = self.outc(x)
        
        ### Steel Work   ####
        
        
        conv1 = self.dconv_down1(img_patches)      
        conv2 = self.dconv_down2(conv1)
        conv3 = self.dconv_down3(conv2)
        conv4 = self.dconv_down4(conv3)
        conv5 = self.dconv_down5(conv4)

            # # ## decoder ####
            
        conv5=self.m(conv5)
        u0=self.dconv_up0(conv5)
        u0 = torch.cat([u0, conv4], dim=1) 

        u1=self.dconv_up1(u0) 
        u1 = torch.cat([u1, conv3], dim=1) 
            
        u2=self.dconv_up2(u1)
        u2=self.up11(u2)
        u2 = torch.cat([u2, conv2], dim=1) 
            
        u3=self.dconv_up3(u2)
        u3=self.up22(u3)
        u3 = torch.cat([u3, conv1], dim=1) 
        u3=self.dconv_up4(u3)
            
        x_rust=self.conv_last(u3)
          
        return self.activation(x_steel) ,self.activation(x_rust)
    
# def model() -> END_1:
#     model = END_1()
#     return model

# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# from torchsummary import summary
# model = model()
# model.to(device=DEVICE,dtype=torch.float)
# summary(model, [(3,224,224), (3,64,64)])
 
