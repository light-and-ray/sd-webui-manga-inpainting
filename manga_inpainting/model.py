from PIL import Image, ImageOps
import torch
import numpy as np
import os
from modules import devices
from .tools import (convertImageIntoPILFormat, convertIntoCNImageFormat, generateSeed,
)
from .repo.src.config import Config
from .repo.src.manga_inpaintor import MangaInpaintor


def genMangaLines(image: Image.Image):
    from scripts.preprocessor.legacy.processor import lineart_anime_denoise
    resoluion = 512
    image = convertIntoCNImageFormat(image)
    image = lineart_anime_denoise(image, resoluion)[0]
    image = convertImageIntoPILFormat(image)
    image = ImageOps.invert(image)
    return image


def process(image: Image.Image, mask: Image.Image):
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'repo', 'checkpoints', 'config.yml')
    config = Config(config_path)

    config.MODEL = 4
    config.INPUT_SIZE = 0
    config.TEST_FLIST = image
    config.TEST_MASK_FLIST = mask.resize(image.size)
    config.TEST_LINE_FLIST = genMangaLines(image).resize(image.size)
    config.RESULTS = None

    config.DEVICE = devices.device
    config.SEED = generateSeed()
    torch.manual_seed(config.SEED)
    torch.cuda.manual_seed_all(config.SEED)
    np.random.seed(config.SEED)
    config.GPU = [0]

    model = MangaInpaintor(config)
    model.load()

    with torch.no_grad():
        result = model.test()

    return result
