import torchvision.models as models
import torch.nn as nn


class VGG19(nn.Module):
    """
    Class for feature extraction for computing style and content loss.
    This class based on pretrained VGG19
    """

    def __init__(self, device, layers_for_style=None, layers_for_content=None):

        """
        Initialization function of backbone for feature extraction
        :param device: Device for loading model, torch cpu, gpu..
        :param layers_for_style: List of layers which features we want to extract for style loss.
        The format as following: first part layer type, 'conv' after that layer's serial index.
         For example conv_1 represents VGG19 conv1_1 layer, conv_5 represents conv3_1 of VGG19
        :param layers_for_content: Similar to style format with only difference that those layers for content
        layers for feature extraction.
        """
        super().__init__()

        # Parameters initialization
        if layers_for_content is None:
            layers_for_content = ['conv_8']
        if layers_for_style is None:
            layers_for_style = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

        self.device = device
        self.layers_for_style=layers_for_style
        self.layers_for_content=layers_for_content

        # Load original VGG19 net with pretrained parameters.
        self.cnn = models.vgg19(weights='VGG19_Weights.DEFAULT', progress=False).features.to(device).eval().requires_grad_(False)

    def forward(self, x):
        """
        Forward pass for a backbone
        :param x:
        :return: list of features maps from layers that was initialized for style and list for content features maps
        """
        list_style_features = []
        list_content_features = []

        layer_ind = 0
        for layer in self.cnn:
            if isinstance(layer, nn.Conv2d):
                layer_ind += 1
                name = 'conv_{}'.format(layer_ind)
            elif isinstance(layer, nn.ReLU):
                name = 'relu_{}'.format(layer_ind)
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = 'pool_{}'.format(layer_ind)
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(layer_ind)
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))
            x = layer(x)

            if name in self.layers_for_content:

                list_content_features.append(x)

            if name in self.layers_for_style:
                list_style_features.append(x)

        return list_style_features, list_content_features


class VGG16(nn.Module):
    """
    Class for feature extraction for computing style and content loss.
    This class based on pretrained VGG16
    """
    def __init__(self, device, layers_for_style=None, layers_for_content=None):
        """
        Initialization function of backbone for feature extraction
        :param device: Device for loading model, torch cpu, gpu..
        :param layers_for_style: List of layers which features we want to extract for style loss.
        The format as following: first part layer type, 'conv', after that layer's serial index.
         For example conv_1 represents VGG16 conv1_1 layer, conv_5 represents conv3_1 of VGG16
        :param layers_for_content: Similar to style format with only difference that those layers for content
        layers for feature extraction.
        """

        super().__init__()
        if layers_for_content is None:
            layers_for_content = ['conv_8']
        if layers_for_style is None:
            layers_for_style = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

        self.device = device
        self.layers_for_style=layers_for_style
        self.layers_for_content=layers_for_content
        self.cnn = models.vgg16(weights='VGG16_Weights.DEFAULT', progress=False).features.to(device).eval().requires_grad_(False)

    def forward(self, x):
        """
        Forward pass for a backbone
        :param x:
        :return: list of features maps from layers that was initialized for style and list for content features maps
        """
        list_style_features = []
        list_content_features = []

        layer_ind = 0
        for layer in self.cnn:
            if isinstance(layer, nn.Conv2d):
                layer_ind += 1
                name = 'conv_{}'.format(layer_ind)
            elif isinstance(layer, nn.ReLU):
                name = 'relu_{}'.format(layer_ind)
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = 'pool_{}'.format(layer_ind)
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(layer_ind)
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))
            x = layer(x)

            if name in self.layers_for_content:

                list_content_features.append(x)

            if name in self.layers_for_style:
                list_style_features.append(x)

        return list_style_features, list_content_features

