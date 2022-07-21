import torch
from torchvision import transforms
from collections import OrderedDict
import torch.nn.functional as F
import torch.nn as nn
import torchvision.models as models

class VGG_PoH(nn.Module):
    def __init__(self, unuse=False) -> None:
        super().__init__()
        self.vgg16 = models.vgg16()
        self.fc1 = nn.Linear(1000, 2048)
        self.fc2 = nn.Linear(2048, 11)
        # poh result
        self.act = nn.Softmax(dim=1)
    def forward(self, x):
        x = self.vgg16(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.act(x)
        return x

def test_forword():
    model = VGG_PoH()
    input_tensor = torch.rand(5,3,64, 64)
    output = model(input_tensor)
    if type(output)==tuple:
        for out in output:
            print(out.shape)
    else:
        print(output.shape)

if __name__ =='__main__':
    test_forword()

