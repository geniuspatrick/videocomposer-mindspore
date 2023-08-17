import math
import random

import numpy as np
from PIL import Image

import mindspore as ms
from mindspore import nn, ops
from mindspore.dataset import vision
from mindspore.dataset.vision import Inter as InterpolationMode

from .autoencoder import DiagonalGaussianDistribution


def beta_schedule(schedule, num_timesteps=1000, init_beta=None, last_beta=None):
    """
    This code defines a function beta_schedule that generates a sequence of beta values based on the given input parameters. These beta values can be used in video diffusion processes. The function has the following parameters:
        schedule(str): Determines the type of beta schedule to be generated. It can be 'linear', 'linear_sd', 'quadratic', or 'cosine'.
        num_timesteps(int, optional): The number of timesteps for the generated beta schedule. Default is 1000.
        init_beta(float, optional): The initial beta value. If not provided, a default value is used based on the chosen schedule.
        last_beta(float, optional): The final beta value. If not provided, a default value is used based on the chosen schedule.
    The function returns a PyTorch tensor containing the generated beta values. The beta schedule is determined by the schedule parameter:
        1.Linear: Generates a linear sequence of beta values between init_beta and last_beta.
        2.Linear_sd: Generates a linear sequence of beta values between the square root of init_beta and the square root of last_beta, and then squares the result.
        3.Quadratic: Similar to the 'linear_sd' schedule, but with different default values for init_beta and last_beta.
        4.Cosine: Generates a sequence of beta values based on a cosine function, ensuring the values are between 0 and 0.999.
    If an unsupported schedule is provided, a ValueError is raised with a message indicating the issue.
    """
    if schedule == "linear":
        scale = 1000.0 / num_timesteps
        init_beta = init_beta or scale * 0.0001
        last_beta = last_beta or scale * 0.02
        return ops.linspace(init_beta, last_beta, num_timesteps).to(ms.float64)
    elif schedule == "linear_sd":
        return ops.linspace(init_beta**0.5, last_beta**0.5, num_timesteps).to(ms.float64) ** 2
    elif schedule == "quadratic":
        init_beta = init_beta or 0.0015
        last_beta = last_beta or 0.0195
        return ops.linspace(init_beta**0.5, last_beta**0.5, num_timesteps).to(ms.float64) ** 2
    elif schedule == "cosine":
        betas = []
        for step in range(num_timesteps):
            t1 = step / num_timesteps
            t2 = (step + 1) / num_timesteps
            fn = lambda u: math.cos((u + 0.008) / 1.008 * math.pi / 2) ** 2
            betas.append(min(1.0 - fn(t2) / fn(t1), 0.999))
        return ms.Tensor(betas, dtype=ms.float64)
    else:
        raise ValueError(f"Unsupported schedule: {schedule}")


def random_resize(img, size):
    img = [
        vision.Resize(
            size,
            interpolation=random.choice(
                [InterpolationMode.BILINEAR, InterpolationMode.BICUBIC, InterpolationMode.ANTIALIAS]
            ),
        )(u)
        for u in img
    ]
    return img


class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        # fast resize
        while min(img.size) >= 2 * self.size:
            img = img.resize((img.width // 2, img.height // 2), resample=Image.BOX)
        scale = self.size / min(img.size)
        img = img.resize((round(scale * img.width), round(scale * img.height)), resample=Image.BICUBIC)

        # center crop
        x1 = (img.width - self.size) // 2
        y1 = (img.height - self.size) // 2
        img = img.crop((x1, y1, x1 + self.size, y1 + self.size))
        return img


class AddGaussianNoise(object):
    def __init__(self, mean=0.0, std=0.1):
        self.std = std
        self.mean = mean

    def __call__(self, img):
        assert isinstance(img, ms.Tensor)
        dtype = img.dtype
        if not img.is_floating_point():
            img = img.to(ms.float32)
        out = img + self.std * ops.randn_like(img) + self.mean
        if out.dtype != dtype:
            out = out.to(dtype)
        return out

    def __repr__(self):
        return self.__class__.__name__ + "(mean={0}, std={1})".format(self.mean, self.std)


def make_masked_images(imgs, masks):
    masked_imgs = []
    for i, mask in enumerate(masks):
        # concatenation
        masked_imgs.append(ops.cat([imgs[i] * (1 - mask), (1 - mask)], axis=1))
    return ops.stack(masked_imgs, axis=0)


# @torch.no_grad()
def get_first_stage_encoding(encoder_posterior):
    scale_factor = 0.18215
    if isinstance(encoder_posterior, DiagonalGaussianDistribution):
        z = encoder_posterior.sample()
    elif isinstance(encoder_posterior, ms.Tensor):
        z = encoder_posterior
    else:
        raise NotImplementedError(f"encoder_posterior of type '{type(encoder_posterior)}' not yet implemented")
    return scale_factor * z


class FrozenOpenCLIPEmbedder(nn.Cell):
    """
    Uses the OpenCLIP transformer encoder for text
    """

    LAYERS = [
        # "pooled",
        "last",
        "penultimate",
    ]

    def __init__(self, arch="ViT-H-14", pretrained="laion2b_s32b_b79k", max_length=77, freeze=True, layer="last"):
        super().__init__()
        assert layer in self.LAYERS

        model, _, _ = open_clip.create_model_and_transforms(arch, pretrained=pretrained)
        del model.visual
        self.model = model

        self.max_length = max_length
        if freeze:
            self.freeze()
        self.layer = layer
        if self.layer == "last":
            self.layer_idx = 0
        elif self.layer == "penultimate":
            self.layer_idx = 1
        else:
            raise NotImplementedError()

    def freeze(self):
        self.model = self.model.set_train(False)
        for param in self.get_parameters():
            param.requires_grad = False

    def construct(self, text):
        tokens = open_clip.tokenize(text)
        z = self.encode_with_transformer(tokens)
        return z

    def encode_with_transformer(self, text):
        x = self.model.token_embedding(text)  # [batch_size, n_ctx, d_model]
        x = x + self.model.positional_embedding
        x = x.transpose(1, 0, 2)  # NLD -> LND
        x = self.text_transformer_forward(x, attn_mask=self.model.attn_mask)
        x = x.transpose(1, 0, 2)  # LND -> NLD
        x = self.model.ln_final(x)
        return x

    def text_transformer_forward(self, x: ms.Tensor, attn_mask=None):
        for i, r in enumerate(self.model.transformer.resblocks):
            if i == len(self.model.transformer.resblocks) - self.layer_idx:
                break
            if self.model.transformer.grad_checkpointing:
                # x = checkpoint(r, x, attn_mask)
                raise NotImplementedError("Gradiant checkpointing is not supported for now!")
            else:
                x = r(x, attn_mask=attn_mask)
        return x

    def encode(self, text):
        return self(text)


class FrozenOpenCLIPVisualEmbedder(nn.Cell):
    """
    Uses the OpenCLIP transformer encoder for text
    """

    LAYERS = [
        # "pooled",
        "last",
        "penultimate",
    ]

    def __init__(
        self,
        arch="ViT-H-14",
        pretrained="laion2b_s32b_b79k",
        max_length=77,
        freeze=True,
        layer="last",
        input_shape=(224, 224, 3),
    ):
        super().__init__()
        assert layer in self.LAYERS

        # version = 'cache/open_clip_pytorch_model.bin'
        model, _, preprocess = open_clip.create_model_and_transforms(arch, pretrained=pretrained)
        del model.transformer
        self.model = model

        data_white = np.ones(input_shape, dtype=np.uint8) * 255
        self.black_image = preprocess(vision.ToPIL()(data_white)).unsqueeze(0)
        self.preprocess = preprocess

        self.max_length = max_length
        if freeze:
            self.freeze()
        self.layer = layer
        if self.layer == "last":
            self.layer_idx = 0
        elif self.layer == "penultimate":
            self.layer_idx = 1
        else:
            raise NotImplementedError()

    def freeze(self):
        self.model = self.model.set_train(False)
        for param in self.get_parameters():
            param.requires_grad = False

    def construct(self, image):
        z = self.model.encode_image(image)
        return z

    def encode(self, text):
        return self(text)


def find_free_port():
    """https://stackoverflow.com/questions/1365265/on-localhost-how-do-i-pick-a-free-port-number"""
    import socket
    from contextlib import closing

    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return str(s.getsockname()[1])


def setup_seed(seed):
    ms.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
