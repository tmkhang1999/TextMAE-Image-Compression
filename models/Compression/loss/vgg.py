import torch.nn as nn
from torchvision import models
from collections import namedtuple


def feature_network(net_type="vgg16", gpu_ids=[], requires_grad=False):
    """
    Define feature network

    Args:
        net_type (str): Type of network, Example: 'vgg16'
        gpu_ids (List[int]): GPU ids
        requires_grad (bool): If True, requires gradient

    Returns:
        feature_network (nn.Module): Feature network

    """
    feature_network = None

    if net_type == "vgg16":
        feature_network = Vgg16(requires_grad=requires_grad)
    else:
        raise NotImplementedError("Feature net name [%s] is not recognized" % net_type)

    feature_network.cuda()

    return feature_network


# feature loss network
class Vgg16(nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg16, self).__init__()
        # os.environ['TORCH_HOME'] = '../../../models/'
        # Load pretrained model
        vgg_pretrained_features = models.vgg16(pretrained=True).features

        # Initialize slice featuremaps
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()

        # Add module for each slice
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        """
        Args:
            X (torch.Tensor): Input tensor

        Returns:
            out (torch.Tensor): Featuremap tensor
        """
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h

        vgg_outputs = namedtuple(
            "VggOutputs", ["relu1_2", "relu2_2", "relu3_3", "relu4_3"]
        )

        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)
        return out
