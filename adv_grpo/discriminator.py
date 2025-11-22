import torch.nn as nn


# def weights_init(m):
#     classname = m.__class__.__name__
#     if classname.find("Conv") != -1:
#         nn.init.normal_(m.weight.data, 0.0, 0.02)
#     elif classname.find("BatchNorm") != -1:
#         nn.init.normal_(m.weight.data, 1.0, 0.02)
#         nn.init.constant_(m.bias.data, 0)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.constant_(m.bias, 0)

class PatchGANDiscriminator(nn.Module):
    pass


class PatchGANImageDiscriminator(PatchGANDiscriminator):
    r"""Defines a PatchGAN discriminator as in Pix2Pix
    References:
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py

    Parameters:
        input_nc (int)  -- the number of channels in input images
        ndf (int)       -- the number of filters in the last conv layer
        n_layers (int)  -- the number of conv layers in the discriminator
        kw (int)        -- the kernel width of convolutional layers in the discriminator
        padw (int)      -- the padding width for convolutional layers
        use_bias (bool) -- whether to include bias in convolutional layers
        norm_type (str) -- type of normalization to apply
        use_gradfix (bool)  -- whether to use gradfix ops to accelerate r1 penalty calculation
    """

    def __init__(
        self,
        input_nc: int = 3,
        ndf: int = 64,
        n_layers: int = 3,
        kw: int = 4,
        padw: int = 1,
        use_bias: bool = False,
        norm_type: str = "BatchNorm",
    ):
        super().__init__()
        if norm_type == "BatchNorm":
            # norm_layer = nn.BatchNorm2d
            norm_layer = nn.InstanceNorm2d
        else:
            raise ValueError(f"Unsupported normalization layer: {norm_type}")

        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2,inplace=False),
        ]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(
                    ndf * nf_mult_prev,
                    ndf * nf_mult,
                    kernel_size=kw,
                    stride=2,
                    padding=padw,
                    bias=use_bias,
                ),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, inplace=False),
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(
                ndf * nf_mult_prev,
                ndf * nf_mult,
                kernel_size=kw,
                stride=1,
                padding=padw,
                bias=use_bias,
            ),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, inplace=False),
        ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)
        ]  # output 1 channel prediction map
        self.main = nn.Sequential(*sequence)

        # Apply weight init
        self.apply(weights_init)

    def forward(self, input):
        """Standard forward."""
        return self.main(input)
