
import torch
import torch.nn as nn
from torchvision.models.video import r3d_18, R3D_18_Weights
from torchvision.models import resnet50, ResNet50_Weights

class BaselineImageClassifier(nn.Module):
    def __init__(self, backbone_name='resnet50', num_classes=1, pretrained=True):
        super(BaselineImageClassifier, self).__init__()
        self.backbone_name = backbone_name
        
        if backbone_name == 'resnet50':
            weights = ResNet50_Weights.DEFAULT if pretrained else None
            self.model = resnet50(weights=weights)
            
            # Replace the classification head
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Linear(num_ftrs, num_classes)
            )
        else:
             raise ValueError(f"Unknown image backbone: {backbone_name}")
             
    def forward(self, x):
        return self.model(x)

class BaselineVideoClassifier(nn.Module):
    def __init__(self, backbone_name='r3d_18', num_classes=1, pretrained=True):
        super(BaselineVideoClassifier, self).__init__()
        self.backbone_name = backbone_name
        
        if backbone_name == 'r3d_18':
            if pretrained:
                weights = R3D_18_Weights.DEFAULT
                self.model = r3d_18(weights=weights)
            else:
                self.model = r3d_18(weights=None)
            
            # Replace the classification head
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Linear(num_ftrs, num_classes)
            )
            
        elif backbone_name == 'simple_cnn':
            self.model = Simple3DCNN(num_classes=num_classes)
        else:
            raise ValueError(f"Unknown backbone: {backbone_name}")

    def forward(self, x):
        return self.model(x)

class Simple3DCNN(nn.Module):
    def __init__(self, num_classes=1):
        super(Simple3DCNN, self).__init__()
        # Input: (B, 3, T, H, W) e.g., (B, 3, 16, 112, 112)
        self.conv1 = nn.Conv3d(3, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn1 = nn.BatchNorm3d(32)
        self.relu = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn2 = nn.BatchNorm3d(64)
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn3 = nn.BatchNorm3d(128)
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        
        # Adaptive pooling to handle varying spatial/temporal sizes
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool3(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
