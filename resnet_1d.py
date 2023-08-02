import torch
import torch.nn as  nn
import torch.nn.functional as F


class Block(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(Block, self).__init__()
       

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=7, padding=3, stride=stride, bias=False)
        self.batch_norm1 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(p=0.25)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=7, padding=3, stride=1, bias=False)
        self.batch_norm2 = nn.BatchNorm1d(out_channels)

        self.i_downsample = i_downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
      identity = x

      out = self.dropout(self.relu(self.batch_norm1(self.conv1(x))))
      
      out = self.batch_norm2(self.conv2(out))

      if self.i_downsample is not None:
          identity = self.i_downsample(x)
      out += identity
      out = self.relu(out)
      return out


        
        
class ResNet(nn.Module):
    def __init__(self, ResBlock, layer_list, num_classes=9, num_channels=12):
        super(ResNet, self).__init__()
        self.in_channels = 64
        
        self.conv1 = nn.Conv1d(num_channels, 64, kernel_size=15, stride=2, padding=7, bias=False)
        self.batch_norm1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool1d(kernel_size = 3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(ResBlock, layer_list[0], planes=64)
        self.layer2 = self._make_layer(ResBlock, layer_list[1], planes=128, stride=2)
        self.layer3 = self._make_layer(ResBlock, layer_list[2], planes=256, stride=2)
        self.layer4 = self._make_layer(ResBlock, layer_list[3], planes=512, stride=2)
        
        self.adaptiveavgpool = nn.AdaptiveAvgPool1d(1)
        self.adaptivemaxpool = nn.AdaptiveMaxPool1d(1)
        
        
        self.fc = nn.Linear(512*ResBlock.expansion*2, num_classes)
        self.dropout = nn.Dropout(0.25)
        
    def forward(self, x):
        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.max_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x1 = self.adaptiveavgpool(x)
        x2 = self.adaptivemaxpool(x)        
        x = torch.cat((x1, x2), dim=1)
        x = x.view(x.size(0), -1)
        return self.fc(x)
        
    def _make_layer(self, ResBlock, blocks, planes, stride=1):
        ii_downsample = None
        layers = []
        
        if stride != 1 or self.in_channels != planes*ResBlock.expansion:
            ii_downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, planes*ResBlock.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm1d(planes*ResBlock.expansion)
            )
            
        layers.append(ResBlock(self.in_channels, planes, i_downsample=ii_downsample, stride=stride))
        self.in_channels = planes*ResBlock.expansion
        
        for i in range(blocks-1):
            layers.append(ResBlock(self.in_channels, planes))
            
        return nn.Sequential(*layers)

def ResNet18(num_classes, channels=12):
    return ResNet(Block, [2,2,2,2], num_classes, channels)

def ResNet34(num_classes, channels=12):
    return ResNet(Block, [3,4,6,3], num_classes, channels)      
        
def ResNet50(num_classes, channels=3):
    return ResNet(Bottleneck, [3,4,6,3], num_classes, channels)
    
def ResNet101(num_classes, channels=3):
    return ResNet(Bottleneck, [3,4,23,3], num_classes, channels)

def ResNet152(num_classes, channels=3):
    return ResNet(Bottleneck, [3,8,36,3], num_classes, channels)