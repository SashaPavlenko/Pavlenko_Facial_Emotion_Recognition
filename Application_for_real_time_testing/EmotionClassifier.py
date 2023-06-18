import torch
import torch.nn as nn

class EmotionClassifier(nn.Module):
    def __init__(self, in_nc=1, nc=32, out_nc=7):
        super(EmotionClassifier, self).__init__()
        self.c1 = nn.Conv2d(in_channels=in_nc, out_channels=nc, kernel_size=3, padding=1)
        self.r1 = nn.ReLU()
        self.c2 = nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=3, padding=1)
        self.r2 = nn.ReLU()
        self.c3 = nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=3, padding=1)
        self.r3 = nn.ReLU()
        self.mp1 = nn.MaxPool2d(2)
        self.bn1 = nn.BatchNorm2d(nc)

        self.c4 = nn.Conv2d(in_channels=nc, out_channels=2*nc, kernel_size=3, padding=1)
        self.r4 = nn.ReLU()
        self.c5 = nn.Conv2d(in_channels=2*nc, out_channels=2*nc, kernel_size=3, padding=1)
        self.r5 = nn.ReLU()
        self.c6 = nn.Conv2d(in_channels=2*nc, out_channels=2*nc, kernel_size=3, padding=1)
        self.r6 = nn.ReLU()
        self.mp2 = nn.MaxPool2d(2)
        self.bn2 = nn.BatchNorm2d(2*nc)
            
        self.c7 = nn.Conv2d(in_channels=2*nc, out_channels=4*nc, kernel_size=3, padding=1)
        self.r7 = nn.ReLU()
        self.c8 = nn.Conv2d(in_channels=4*nc, out_channels=4*nc, kernel_size=3, padding=1)
        self.r8 = nn.ReLU()
        self.c9 = nn.Conv2d(in_channels=4*nc, out_channels=4*nc, kernel_size=3, padding=1)
        self.r9 = nn.ReLU()
        self.mp3 = nn.MaxPool2d(2)
        self.bn3 = nn.BatchNorm2d(4*nc)
        
        self.c10 = nn.Conv2d(in_channels=4*nc, out_channels=2*nc, kernel_size=3, padding=1)
        self.r10 = nn.ReLU()
        self.c11 = nn.Conv2d(in_channels=2*nc, out_channels=2*nc, kernel_size=3, padding=1)
        self.r11 = nn.ReLU()
        self.c12 = nn.Conv2d(in_channels=2*nc, out_channels=2*nc, kernel_size=3, padding=1)
        self.r12 = nn.ReLU()
        self.mp4 = nn.MaxPool2d(2)
        self.bn4 = nn.BatchNorm2d(2*nc)

        self.f =  nn.Flatten()
        self.fc1 = nn.Linear(int(48*48*2**(int(math.log2(nc))-7)), 2*nc)
        self.fc2 = nn.Linear(2*nc, 2*nc)
        self.fc3 = nn.Linear(2*nc, out_nc)
            
            
        
    def forward(self, features):
        out = self.r1(self.c1(features))        
        out = self.r2(self.c2(out))        
        out = self.r3(self.c3(out))        
        out = self.mp1(out)        
        out = self.bn1(out)
        out = self.r4(self.c4(out))
        out = self.r5(self.c5(out))
        out = self.r6(self.c6(out))
        out = self.mp2(out)
        out = self.bn2(out)
        
        out = self.r7(self.c7(out))
        out = self.r8(self.c8(out))
        out = self.r9(self.c9(out))
        out = self.mp3(out)
        out = self.bn3(out)
        
        out = self.r10(self.c10(out))
        out = self.r11(self.c11(out))
        out = self.r12(self.c12(out))
        out = self.mp4(out)
        out = self.bn4(out)

        out = self.f(out)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)

        return out
