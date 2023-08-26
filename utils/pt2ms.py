import os

import numpy as np

import mindspore as ms
from mindspore import nn


def get_pt2ms_mappings(model: nn.Cell):
    mappings = {}  # pt_param_name: (ms_param_name, pt_param_to_ms_param_func)

    def check_key(k):
        if k in mappings:
            raise KeyError(f"param name {k} is already in mapping!")

    for name, cell in model.cells_and_names():
        if isinstance(cell, nn.Conv1d):
            check_key(f"{name}.weight")
            mappings[f"{name}.weight"] = f"{name}.weight", lambda x: np.expand_dims(x, axis=-2)
        elif isinstance(cell, nn.Embedding):
            check_key(f"{name}.weight")
            mappings[f"{name}.weight"] = f"{name}.embedding_table", lambda x: x
        elif isinstance(cell, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
            check_key(f"{name}.weight")
            mappings[f"{name}.weight"] = f"{name}.gamma", lambda x: x
            check_key(f"{name}.bias")
            mappings[f"{name}.bias"] = f"{name}.beta", lambda x: x
    return mappings


def convert_state_dict(model, state_dict_pt):
    mappings = get_pt2ms_mappings(model)
    state_dict_ms = {}
    for name_pt, data_pt in state_dict_pt.items():
        name_ms, data_mapping = mappings.get(name_pt, (name_pt, lambda x: x))
        data_ms = data_mapping(data_pt)
        state_dict_ms[name_ms] = ms.Parameter(data_ms.astype(np.float32), name=name_ms)
    return state_dict_ms


def load_pt_weights_in_model(model, checkpoint_file_pt, state_dict_refiners=None):
    checkpoint_file_ms = f"{os.path.splitext(checkpoint_file_pt)[0]}.ckpt"
    if not os.path.exists(checkpoint_file_ms):  # try to load weights from intermediary numpy file.
        checkpoint_file_np = f"{os.path.splitext(checkpoint_file_pt)[0]}.npy"
        if not os.path.exists(checkpoint_file_np):
            raise FileNotFoundError(f"You need to manually convert {checkpoint_file_pt} to {checkpoint_file_np}")
        sd_original = np.load(checkpoint_file_np, allow_pickle=True).item()
        # refine state dict of pytorch
        sd_refined = sd_original
        if state_dict_refiners:
            for refine_fn in state_dict_refiners:
                sd_refined = refine_fn(sd_refined)
        # convert state_dict from pytorch to mindspore
        sd = convert_state_dict(model, sd_refined)
        # save converted state_dict as cache
        ms.save_checkpoint([{"name": k, "data": v} for k, v in sd.items()], checkpoint_file_ms)
    else:  # directly load weights from cached mindspore file.
        sd = ms.load_checkpoint(checkpoint_file_ms)

    param_not_load, ckpt_not_load = ms.load_param_into_net(model, sd, strict_load=True)
    if param_not_load or ckpt_not_load:
        print(f"{param_not_load} in network is not loaded or {ckpt_not_load} in checkpoint is not loaded!")
