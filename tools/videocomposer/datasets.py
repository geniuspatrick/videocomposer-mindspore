import os
import random

import cv2
import imageio
import numpy as np
from PIL import Image

import mindspore as ms
from mindspore import ops
from mindspore.dataset.vision import Inter, Resize

import utils.logging as logging

from ..annotator.mask import make_irregular_mask, make_rectangle_mask, make_uncrop
from ..annotator.motion import extract_motion_vectors

logger = logging.get_logger(__name__)


class VideoDataset(object):
    def __init__(
        self,
        cfg,
        tokenizer=None,
        max_words=30,
        feature_framerate=1,
        max_frames=16,
        image_resolution=224,
        transforms=None,
        mv_transforms=None,
        misc_transforms=None,
        vit_transforms=None,
        vit_image_size=336,
        misc_size=384,
    ):
        self.cfg = cfg

        self.tokenizer = tokenizer
        self.max_words = max_words
        self.feature_framerate = feature_framerate
        self.max_frames = max_frames
        self.image_resolution = image_resolution
        self.transforms = transforms
        self.vit_transforms = vit_transforms
        self.vit_image_size = vit_image_size
        self.misc_transforms = misc_transforms
        self.misc_size = misc_size

        self.mv_transforms = mv_transforms

        self.video_cap_pairs = [[self.cfg.input_video, self.cfg.input_text_desc]]
        self.Vit_image_random_resize = Resize((vit_image_size, vit_image_size), interpolation=Inter.BILINEAR)

        self.SPECIAL_TOKEN = {
            "CLS_TOKEN": "<|startoftext|>",
            "SEP_TOKEN": "<|endoftext|>",
            "MASK_TOKEN": "[MASK]",
            "UNK_TOKEN": "[UNK]",
            "PAD_TOKEN": "[PAD]",
        }  # TODO: get it from the tokenizer.special_tokens

    def __len__(self):
        return len(self.video_cap_pairs)

    def __getitem__(self, index):
        video_key, cap_txt = self.video_cap_pairs[index]

        total_frames = None

        feature_framerate = self.feature_framerate
        if os.path.exists(video_key):
            try:
                ref_frame, vit_image, video_data, misc_data, mv_data = self._get_video_traindata(
                    video_key, feature_framerate, total_frames, self.cfg.mvs_visual
                )
            except Exception as e:
                print("{} get frames failed... with error: {}".format(video_key, e), flush=True)

                ref_frame = ops.zeros((3, self.vit_image_size, self.vit_image_size))
                # vit_image = ops.zeros((3, self.vit_image_size, self.vit_image_size))
                video_data = ops.zeros((self.max_frames, 3, self.image_resolution, self.image_resolution))
                misc_data = ops.zeros((self.max_frames, 3, self.misc_size, self.misc_size))

                mv_data = ops.zeros((self.max_frames, 2, self.image_resolution, self.image_resolution))
        else:
            print("The video path does not exist or no video dir provided!")
            ref_frame = ops.zeros((3, self.vit_image_size, self.vit_image_size))
            # vit_image = ops.zeros((3, self.vit_image_size, self.vit_image_size))
            video_data = ops.zeros((self.max_frames, 3, self.image_resolution, self.image_resolution))
            misc_data = ops.zeros((self.max_frames, 3, self.misc_size, self.misc_size))

            mv_data = ops.zeros((self.max_frames, 2, self.image_resolution, self.image_resolution))

        # inpainting mask
        p = random.random()
        if p < 0.7:
            mask = make_irregular_mask(512, 512)
        elif p < 0.9:
            mask = make_rectangle_mask(512, 512)
        else:
            mask = make_uncrop(512, 512)
        mask = ms.Tensor(
            cv2.resize(mask, (self.misc_size, self.misc_size), interpolation=cv2.INTER_NEAREST), ms.float32
        ).unsqueeze(0)

        mask = ops.repeat_interleave(mask.unsqueeze(0), repeats=self.max_frames, axis=0)

        return ref_frame, cap_txt, video_data, misc_data, feature_framerate, mask, mv_data

    def _get_video_traindata(self, video_key, feature_framerate, total_frames, visual_mv):
        filename = video_key
        for _ in range(5):
            try:
                frame_types, frames, mvs, mvs_visual = extract_motion_vectors(
                    input_video=filename, fps=feature_framerate, visual_mv=visual_mv
                )
                # os.remove(filename)
                break
            except Exception as e:
                print("{} read video frames and motion vectors failed with error: {}".format(video_key, e), flush=True)

        total_frames = len(frame_types)
        start_indexs = np.where(
            (np.array(frame_types) == "I") & (total_frames - np.arange(total_frames) >= self.max_frames)
        )[0]
        start_index = np.random.choice(start_indexs)
        indices = np.arange(start_index, start_index + self.max_frames)

        # note frames are in BGR mode, need to trans to RGB mode
        frames = [Image.fromarray(frames[i][:, :, ::-1]) for i in indices]
        mvs = [ms.Tensor(mvs[i].permute(2, 0, 1)) for i in indices]
        mvs = ops.stack(mvs)
        # set_trace()
        # if mvs_visual != None:
        if visual_mv:
            # images = [(mvs_visual[i][:,:,::-1]*255).astype('uint8') for i in indices]
            images = [(mvs_visual[i][:, :, ::-1]).astype("uint8") for i in indices]
            # images = [mvs_visual[i] for i in indices]
            # images = [(image.numpy()*255).astype('uint8') for image in images]
            path = self.cfg.log_dir + "/visual_mv/" + video_key.split("/")[-1] + ".gif"
            if not os.path.exists(self.cfg.log_dir + "/visual_mv/"):
                os.makedirs(self.cfg.log_dir + "/visual_mv/", exist_ok=True)
            print("save motion vectors visualization to :", path)
            imageio.mimwrite(path, images, fps=8)

        # mvs_visual = [torch.from_numpy(mvs_visual[i].transpose((2,0,1))) for i in indices]
        # mvs_visual = torch.stack(mvs_visual)
        # mvs_visual = self.mv_transforms(mvs_visual)

        have_frames = len(frames) > 0
        middle_indix = int(len(frames) / 2)
        if have_frames:
            ref_frame = frames[middle_indix]
            vit_image = self.vit_transforms(ref_frame)
            misc_imgs_np = self.misc_transforms[:2](frames)
            misc_imgs = self.misc_transforms[2:](misc_imgs_np)
            frames = self.transforms(frames)
            mvs = self.mv_transforms(mvs)
        else:
            # ref_frame = Image.fromarray(np.zeros((3, self.image_resolution, self.image_resolution)))
            vit_image = ops.zeros((3, self.vit_image_size, self.vit_image_size))

        video_data = ops.zeros((self.max_frames, 3, self.image_resolution, self.image_resolution))
        mv_data = ops.zeros((self.max_frames, 2, self.image_resolution, self.image_resolution))
        misc_data = ops.zeros((self.max_frames, 3, self.misc_size, self.misc_size))
        if have_frames:
            video_data[: len(frames), ...] = frames  # [[XX...],[...], ..., [0,0...], [], ...]
            misc_data[: len(frames), ...] = misc_imgs
            mv_data[: len(frames), ...] = mvs

        ref_frame = vit_image

        del frames
        del misc_imgs
        del mvs

        return ref_frame, vit_image, video_data, misc_data, mv_data
