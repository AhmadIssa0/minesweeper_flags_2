

import torch
import torch.nn as nn


class SkipBlock(nn.Module):
    def __init__(self, nn_block: nn.Module):
        """

        :param nn_block: a neural network preserving the shape of the input tensor.
        """
        super().__init__()
        self._nn_block = nn_block

    def forward(self, x):
        return x + self._nn_block(x)


class ConvNeXtBlock(nn.Module):
    def __init__(self, channels):
        super(ConvNeXtBlock, self).__init__()


        # depthwise convolution
        self.dwconv = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=(7, 7), stride=(1, 1),
                                padding='same', groups=channels)
        self.layernorm = nn.LayerNorm(normalized_shape=channels)
        # 1x1 (pointwise) convolution implemented as linear layer (since we're going to permute dimensions)
        self.pwconv1 = nn.Linear(channels, 4 * channels)
        self.relu = nn.ReLU()
        self.pwconv2 = nn.Linear(4 * channels, channels)

    def forward(self, x):
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.layernorm(x)
        x = self.pwconv1(x)
        x = self.relu(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        return x


class ConvNeXt(nn.Module):
    def __init__(self, input_channels, output_channels, intermediary_channels, n_blocks):
        super().__init__()
        modules = [nn.Conv2d(in_channels=input_channels, out_channels=intermediary_channels, kernel_size=(1, 1),
                             stride=(1, 1), padding='same', groups=1)]

        for i in range(n_blocks):
            modules.append(SkipBlock(ConvNeXtBlock(intermediary_channels)))

        modules.append(nn.Conv2d(in_channels=intermediary_channels, out_channels=output_channels, kernel_size=(1, 1),
                                 stride=(1, 1), padding='same', groups=1))
        self.model = nn.Sequential(*modules)

    def forward(self, x):
        return self.model(x)


class Resnet(nn.Module):
    def __init__(self, input_channels, n_channels=20, n_blocks=5):
        super(Resnet, self).__init__()
        module_list = [nn.Conv2d(in_channels=input_channels, out_channels=n_channels, kernel_size=(3, 3), stride=(1, 1),
                                 padding='same', groups=1)]

        for i in range(n_blocks):
            module_list.append(SkipBlock(
                nn.Sequential(
                    nn.ReLU(),
                    nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=(3, 3), stride=(1, 1),
                              padding='same', groups=1),
                    nn.ReLU(),
                    nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=(3, 3), stride=(1, 1),
                              padding='same', groups=1),
                )
            ))

        module_list.append(nn.Conv2d(in_channels=20, out_channels=1, kernel_size=(1, 1), stride=(1, 1),
                                     padding='same', groups=1))

        self._model = nn.Sequential(*module_list)

    def forward(self, x):
        return self._model(x)
