import logging

from tools.videocomposer.inference_multi import inference_multi
from tools.videocomposer.inference_single import inference_single
from utils.config import Config


def main():
    """
    Main function to spawn the train and test process.
    """
    cfg = Config(load=True)
    if hasattr(cfg, "TASK_TYPE") and cfg.TASK_TYPE == "MULTI_TASK":
        logging.info("TASK TYPE: %s " % cfg.TASK_TYPE)
        inference_multi(cfg.cfg_dict)
    elif hasattr(cfg, "TASK_TYPE") and cfg.TASK_TYPE == "SINGLE_TASK":
        logging.info("TASK TYPE: %s " % cfg.TASK_TYPE)
        inference_single(cfg.cfg_dict)
    else:
        logging.info("Not support task %s" % cfg.TASK_TYPE)


if __name__ == "__main__":
    main()
