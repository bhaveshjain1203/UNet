import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

class Conv(nn.Module):
    def __init__(self,inChannels,outChannels):
        super(Conv,self).__init__()
        
        # Creating a sequential model with 2D convolution
        ''' padding = 1 for same convolution i.e height and width of input will be same after convolution
            TODO bias = false is set as batchnorm is used ,why?
        '''
        self.conv = nn.Sequential(
            nn.Conv2d(inChannels, outChannels,kernel_size = 3, stride = 1,padding = 1, bias = False),
            nn.BatchNorm2d(outChannels),
            nn.ReLU(inplace=True),
            nn.Conv2d(outChannels, outChannels, kernel_size = 3, stride = 1, padding = 1, bias=False),
            nn.BatchNorm2d(outChannels),
            nn.ReLU(inplace=True),
        )
        
    def forward(self, x):
        return self.conv(x)
    
# inChannels = 3 -> RGB
class Unet(nn.Module):
    def __init__(self, inChannels = 3, outChannels = 1,features = [64,128,256,512]): # TODO OUTchannles
        super(Unet,self).__init__()

        self.down = nn.ModuleList()
        self.up = nn.ModuleList()

        self.pool = nn.MaxPool2d(kernel_size=2,stride=2)

        # ENCODER
        for feature in features:
            self.down.append(Conv(inChannels=inChannels,outChannels=feature))
            inChannels = feature

        # DECODER
        # Why transposeConvolution is used for upsampling instead of using linear and then conv layer 
        for feature in reversed(features):
            self.up.append(nn.ConvTranspose2d(
                in_channels=feature * 2,out_channels=feature,kernel_size=2,stride=2
            ))

            self.up.append(Conv(inChannels=feature * 2,outChannels=feature))


        self.bottleneck = Conv(inChannels=features[-1],outChannels=features[-1] * 2)

        self.finalConv = nn.Conv2d(features[0],out_channels=outChannels,kernel_size=1)

    def forward(self,x):
        skip_connections = []

        for down in self.down:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0,len(self.up),2):
            x = self.up[idx](x)
            skip_connection = skip_connections[idx // 2]
            
            if x.shape != skip_connection.shape:
                #just taking height and width,skipping batch size and no of channels
                x = TF.resize(x,size = skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection,x),dim = 1)
            x = self.up[idx + 1](concat_skip)

        return self.finalConv(x)
    
def test():
    x = torch.randn((3,1,160,160))
    model = Unet(inChannels=1,outChannels=1)
    preds = model(x)
    print(x.shape)
    print(preds.shape)
    assert x.shape == preds.shape

if __name__ == "__main__":
    test()









