from PIL import Image, ImageOps
import copy
from typing import Any
from dataclasses import dataclass
from modules import shared
from .model import process
from .tools import (crop, uncrop, areImagesTheSame, applyMaskBlur
)


@dataclass
class CacheData:
    image: Any
    mask: Any
    invert: Any
    padding: Any
    seed: Any
    blur: Any
    result: Any

cachedData = None




def mangaInpaint(image: Image, mask: Image, invert: int, padding: int|None, seed: int, blur: int):
    global cachedData
    result = None
    if cachedData is not None and\
            cachedData.invert == invert and\
            cachedData.padding == padding and\
            cachedData.seed == seed and\
            cachedData.blur == blur and\
            areImagesTheSame(cachedData.image, image) and\
            areImagesTheSame(cachedData.mask, mask):
        result = copy.copy(cachedData.result)
        print("manga inpainted restored from cache")
        shared.state.assign_current_image(result)
    else:
        forCache = CacheData(image.copy(), mask.copy(), invert, padding, seed, blur, None)
        if invert == 1:
            mask = ImageOps.invert(mask)
        mask = applyMaskBlur(mask, blur)
        initImage = copy.copy(image)
        image = copy.copy(initImage)
        if padding is not None:
            maskNotCropped = mask
            image = crop(image, maskNotCropped, padding)
            mask = crop(mask, maskNotCropped, padding)
        shared.state.textinfo = "manga inpainting"
        tmpImage = process(image, mask, seed)
        inpaintedImage = image.copy()
        inpaintedImage.paste(tmpImage, mask)
        if padding is not None:
            result = uncrop(inpaintedImage, initImage, maskNotCropped, padding)
        shared.state.textinfo = ""
        forCache.result = result.copy()
        cachedData = forCache
        print("manga inpainted cached")

    return result
