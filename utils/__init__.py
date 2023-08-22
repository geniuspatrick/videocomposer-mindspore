import os.path as osp


def DOWNLOAD_TO_CACHE(file_or_dirname=None, cache_dir=None):
    r"""Download OSS [file or folder] to the cache folder.
    Only the 0th process on each node will run the downloading.
    Barrier all processes until the downloading is completed.
    """
    if cache_dir is None:
        cache_dir = osp.join("/".join(osp.abspath(__file__).split("/")[:-2]), "model_weights")
    # source and target paths
    base_path = osp.join(cache_dir, file_or_dirname)

    return base_path
