import math
import random
from typing import Optional

import numpy as np
from PIL import Image

import mindspore as ms
from mindspore import nn, ops
from mindspore.dataset import vision
from mindspore.dataset.vision import Inter as InterpolationMode

from ..clip import CLIPImageProcessor, CLIPModel, CLIPTokenizer, parse, support_list
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


def setup_seed(seed):
    ms.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def load_clip_model(arch, pretrained_ckpt_path):
    """
    Load CLIP model.

    Args:
        arch (str): Model architecture.
        pretrained_ckpt_path (str): Path of the pretrained checkpoint.
    Returns:
        model (CLIPModel): CLIP model.
    """
    if arch.lower() not in support_list:
        raise ValueError(f"arch {arch} is not supported")
    config_path = support_list[arch.lower()]

    config = parse(config_path, pretrained_ckpt_path)
    model = CLIPModel(config)
    return model


def load_ckpt_tokenizer(tokenizer_path):
    text_processor = CLIPTokenizer(tokenizer_path, pad_token="!")
    return text_processor


class FrozenOpenCLIPEmbedder(nn.Cell):
    def __init__(
        self,
        arch="open_clip_vit_h_14",
        pretrained_ckpt_path="./vit-h-14-laion-2b/open_clip_vit_h_14.ckpt",
        tokenizer_path="./vit-h-14-laion-2b/bpe_simple_vocab_16e6.txt.gz",
        freeze=True,
        layer="penultimate",
    ):
        super().__init__()
        model = load_clip_model(arch, pretrained_ckpt_path)
        del model.visual
        self.model = model
        self.layer = layer
        self.freeze = freeze
        if self.freeze:
            self.model.set_train(False)
            for name, param in self.model.parameters_and_names():
                param.requires_grad = False

        if self.layer == "last":
            layer_index = 0
        elif self.layer == "penultimate":
            layer_index = 1
            old_layers = len(self.model.transformer.resblocks)
            self.delete_last_n_layers_from_resblocks(layer_index)
            new_layers = len(self.model.transformer.resblocks)
            print(f"Transformer Resblocks Layers change from {old_layers} to {new_layers}")
        else:
            raise ValueError(f"layer {layer} is not supported")

        self.tokenizer = load_ckpt_tokenizer(tokenizer_path)

    def delete_last_n_layers_from_resblocks(self, layer_index):
        assert layer_index < len(self.model.transformer.resblocks) and layer_index >= 0
        N = len(self.model.transformer.resblocks)
        index = N - 1
        for _ in range(layer_index):
            del self.model.transformer.resblocks[index]
            index -= 1
        return

    def process_text(self, text_prompt):
        return ms.Tensor(self.tokenizer(text_prompt, padding="max_length", max_length=77)["input_ids"])

    def construct(self, text):
        if isinstance(text, str):
            text = [text]
        token_ids = self.process_text(text)
        text_features = self.get_text_features(token_ids)
        return text_features

    def encode(self, text):
        return self.construct(text)

    def get_text_features(self, text: ms.Tensor, input_ids: Optional[ms.Tensor] = None):
        r"""Get_text_features

        Args:
            text (ms.Tensor): A text id tensor processed by tokenizer.
            input_ids (Optional[ms.Tensor]): Equal to "text",
                if "input_ids" is set, "text" is useless.

        Returns:
            Text feature.
        """
        if input_ids is not None:
            text = input_ids
        text_ = self.model.token_embedding(text)
        text_ = text_.astype(self.model.dtype)
        text_ = ops.Add()(text_, self.model.positional_embedding)
        text_ = text_.transpose(1, 0, 2)
        text_ = self.model.transformer(text_)
        text_ = text_.transpose(1, 0, 2)
        text_ = self.model.ln_final(text_)

        return text_


class FrozenOpenCLIPVisualEmbedder(nn.Cell):
    def __init__(
        self,
        arch="open_clip_vit_h_14",
        pretrained_ckpt_path="./vit-h-14-laion-2b/open_clip_vit_h_14.ckpt",
        freeze=True,
        layer="penultimate",
        resolution=224,
    ):
        super().__init__()
        model = load_clip_model(arch, pretrained_ckpt_path)
        del model.transformer

        self.model = model
        self.image_processor = CLIPImageProcessor(resolution)

        data_white = np.ones((resolution, resolution, 3)) * 255
        self.black_image = Image.fromarray(data_white.astype(np.uint8)).convert("RGB")

        self.layer = layer  # the layer does not apply to visual embedder
        if self.layer == "last":
            self.layer_index = 0
        elif self.layer == "penultimate":
            self.layer_index = 1
        else:
            raise ValueError(f"layer {layer} is not supported")

        self.freeze = freeze
        if self.freeze:
            self.model.set_train(False)
            for name, param in self.model.parameters_and_names():
                param.requires_grad = False

    def construct(self, image):
        if not isinstance(image, list):
            image_ = self.image_processor(image)
            image_features = self.model.get_image_features(image_)
        else:
            image_ = [self.image_processor(img) for img in image]
            image_features = [self.model.get_image_features(img) for img in image_]
        # the returned features are non-normalilzed

        # normalization
        # if not is_old_ms_version("2.0.0-alpha"):
        #     L2_norm_ops = partial(ops.norm, ord=2, dim=1, keepdim=True)
        # else:
        #     L2_norm_ops = partial(ops.norm, p=2, axis=1, keep_dims=True)

        # image_features = L2_norm_ops(image_features) if not isinstance(image_features, list) else [
        #     L2_norm_ops(img_feat) for img_feat in image_features]
        return image_features
