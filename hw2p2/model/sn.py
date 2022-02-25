import torch
import torch.nn as nn
import torch.nn.functional as F


class Network(nn.Module):
    """
    The Very Low early deadline architecture is a 4-layer CNN.
    The first Conv layer has 64 channels, kernel size 7, and stride 4.
    The next three have 128, 256, and 512 channels. Each have kernel size 3 and stride 2.
    Think about what the padding should be for each layer to not change spatial resolution.
    Each Conv layer is accompanied by a Batchnorm and ReLU layer.
    Finally, you want to average pool over the spatial dimensions to reduce them to 1 x 1.
    Then, remove (Flatten?) these trivial 1x1 dimensions away.
    Look through https://pytorch.org/docs/stable/nn.html 
    TODO: Fill out the model definition below! 

    Why does a very simple network have 4 convolutions?
    Input images are 224x224. Note that each of these convolutions downsample.
    Downsampling 2x effectively doubles the receptive field, increasing the spatial
    region each pixel extracts features from. Downsampling 32x is standard
    for most image models.

    Why does a very simple network have high channel sizes?
    Every time you downsample 2x, you do 4x less computation (at same channel size).
    To maintain the same level of computation, you 2x increase # of channels, which 
    increases computation by 4x. So, balances out to same computation.
    Another intuition is - as you downsample, you lose spatial information. Want
    to preserve some of it in the channel dimension.
    """
    def __init__(self, num_classes=7000):
        super().__init__()

        self.backbone = nn.Sequential(
            # Note that first conv is stride 4. It is (was?) standard to downsample.
            # 4x early on, as with 224x224 images, 4x4 patches are just low-level details.
            # Food for thought: Why is the first conv kernel size 7, not kernel size 3?

            nn.Conv2d(3, 64, 7, stride=4, bias=False), # TODO: Conv group 1
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, 3, stride=2, bias=False), # TODO: Conv group 2
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 256, 3, stride=2, bias=False), # TODO: Conv group 3
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 512, 3, stride=2, bias=False), # TODO: Conv group 4
            nn.BatchNorm2d(512),
            nn.ReLU(),  

            nn.AvgPool2d(6), # TODO: Average pool over & reduce the spatial dimensions to (1, 1)
            nn.Flatten() # TODO: Collapse (Flatten) the trivial (1, 1) dimensions
            ) 
        
        self.cls_layer = nn.Linear(512, num_classes)
    
    def forward(self, x, return_feats=False):
        """
        What is return_feats? It essentially returns the second-to-last-layer
        features of a given image. It's a "feature encoding" of the input image,
        and you can use it for the verification task. You would use the outputs
        of the final classification layer for the classification task.

        You might also find that the classification outputs are sometimes better
        for verification too - try both.
        """
        feats = self.backbone(x)
        out = self.cls_layer(feats)

        if return_feats:
            return feats
        else:
            return out