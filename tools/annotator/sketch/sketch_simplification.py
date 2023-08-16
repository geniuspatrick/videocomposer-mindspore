r"""MindSpore re-implementation adapted from the Lua code in ``https://github.com/bobbens/sketch_simplification''.
"""
import os

import mindspore as ms
import mindspore.nn as nn

__all__ = ["SketchSimplification", "sketch_simplification_gan"]


class conv_nd(nn.Cell):
    def __init__(self, dims, *args, **kwargs):
        super().__init__()
        if dims == 1:
            self.conv = nn.Conv1d(*args, **kwargs)
        elif dims == 2:
            self.conv = nn.Conv2d(*args, **kwargs)
        elif dims == 3:
            self.conv = nn.Conv3d(*args, **kwargs)
        else:
            raise ValueError(f"unsupported dimensions: {dims}")

    def construct(self, x, emb=None, context=None):
        x = self.conv(x)
        return x


class SketchSimplification(nn.Cell):
    r"""NOTE:
    1. Input image should has only one gray channel.
    2. Input image size should be divisible by 8.
    3. Sketch in the input/output image is in dark color while background in light color.
    """

    def __init__(self, mean, std):
        assert isinstance(mean, float) and isinstance(std, float)
        super(SketchSimplification, self).__init__()
        self.mean = mean
        self.std = std

        # layers
        self.layers = nn.SequentialCell(
            conv_nd(2, 1, 48, 5, 2, padding=2, has_bias=True, pad_mode="pad"),
            nn.ReLU(),
            conv_nd(2, 48, 128, 3, 1, padding=1, has_bias=True, pad_mode="pad"),
            nn.ReLU(),
            conv_nd(2, 128, 128, 3, 1, padding=1, has_bias=True, pad_mode="pad"),
            nn.ReLU(),
            conv_nd(2, 128, 128, 3, 2, padding=1, has_bias=True, pad_mode="pad"),
            nn.ReLU(),
            conv_nd(2, 128, 256, 3, 1, padding=1, has_bias=True, pad_mode="pad"),
            nn.ReLU(),
            conv_nd(2, 256, 256, 3, 1, padding=1, has_bias=True, pad_mode="pad"),
            nn.ReLU(),
            conv_nd(2, 256, 256, 3, 2, padding=1, has_bias=True, pad_mode="pad"),
            nn.ReLU(),
            conv_nd(2, 256, 512, 3, 1, padding=1, has_bias=True, pad_mode="pad"),
            nn.ReLU(),
            conv_nd(2, 512, 1024, 3, 1, padding=1, has_bias=True, pad_mode="pad"),
            nn.ReLU(),
            conv_nd(2, 1024, 1024, 3, 1, padding=1, has_bias=True, pad_mode="pad"),
            nn.ReLU(),
            conv_nd(2, 1024, 1024, 3, 1, padding=1, has_bias=True, pad_mode="pad"),
            nn.ReLU(),
            conv_nd(2, 1024, 1024, 3, 1, padding=1, has_bias=True, pad_mode="pad"),
            nn.ReLU(),
            conv_nd(2, 1024, 512, 3, 1, padding=1, has_bias=True, pad_mode="pad"),
            nn.ReLU(),
            conv_nd(2, 512, 256, 3, 1, padding=1, has_bias=True, pad_mode="pad"),
            nn.ReLU(),
            nn.Conv2dTranspose(256, 256, 4, 2, padding=1, has_bias=True, pad_mode="pad"),
            nn.ReLU(),
            conv_nd(2, 256, 256, 3, 1, padding=1, has_bias=True, pad_mode="pad"),
            nn.ReLU(),
            conv_nd(2, 256, 128, 3, 1, padding=1, has_bias=True, pad_mode="pad"),
            nn.ReLU(),
            nn.Conv2dTranspose(128, 128, 4, 2, padding=1, has_bias=True, pad_mode="pad"),
            nn.ReLU(),
            conv_nd(2, 128, 128, 3, 1, padding=1, has_bias=True, pad_mode="pad"),
            nn.ReLU(),
            conv_nd(2, 128, 48, 3, 1, padding=1, has_bias=True, pad_mode="pad"),
            nn.ReLU(),
            nn.Conv2dTranspose(48, 48, 4, 2, padding=1, has_bias=True, pad_mode="pad"),
            nn.ReLU(),
            conv_nd(2, 48, 24, 3, 1, padding=1, has_bias=True, pad_mode="pad"),
            nn.ReLU(),
            conv_nd(2, 24, 1, 3, 1, padding=1, has_bias=True, pad_mode="pad"),
            nn.Sigmoid(),
        )

    def construct(self, x):
        r"""x: [B, 1, H, W] within range [0, 1]. Sketch pixels in dark color."""
        x = (x - self.mean) / self.std
        return self.layers(x)


def sketch_simplification_gan(pretrained=False, ckpt_path=None):
    model = SketchSimplification(mean=0.9664114577640158, std=0.0858381272736797)
    if pretrained:
        if ckpt_path is None:
            ckpt_path = os.path.join(os.path.dirname(__file__), "./model_weights/sketch_simplification_gan_ms.ckpt")
        state = ms.load_checkpoint(ckpt_path)

        for pname, p in model.parameters_and_names():
            if p.name != pname and (p.name not in state and pname in state):
                param = state.pop(pname)
                state[p.name] = param  # classifier.conv.weight -> weight; classifier.conv.bias -> bias
        param_not_load, _ = ms.load_param_into_net(model, state)
        if len(param_not_load):
            print("Params not load: {}".format(param_not_load))
    return model


if __name__ == "__main__":
    model = sketch_simplification_gan(pretrained=False)
