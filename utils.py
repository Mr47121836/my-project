import os.path
import time

import torch
import torchvision.models as models
import logging
import sys


logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging


def save_checkpoint(model, optimizer, learning_rate, iteration, checkpoint_path):
    logger.info("Saving model and optimizer state at iteration {} to {}".format(
        iteration, checkpoint_path))
    if not os.path.exists(checkpoint_path):
        os.makedirs("logs")
    model_name = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    path = os.path.join("logs", model_name + '.pth')
    torch.save(model.state_dict(), path)


def load_checkpoint():
    print("权重导入")


if __name__ == '__main__':
    print("此文件用于保存权重")

