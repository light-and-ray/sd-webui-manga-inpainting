from modules import shared
from modules.processing import StableDiffusionProcessingImg2Img
import gradio as gr

ORIGINAL_NAME = '1x_NMKD-YandereInpaint_375000_G'




def getYandereInpaintUpscaler(p: StableDiffusionProcessingImg2Img = None):
    if hasattr(p, 'override_settings'):
        overriden = p.override_settings.get("yandere_inpaint_upscaler", None)
        if overriden:
            return overriden
    res = shared.opts.data.get("yandere_inpaint_upscaler", "ESRGAN_4x")
    return res


def getResolution(p: StableDiffusionProcessingImg2Img = None):
    if hasattr(p, 'override_settings'):
        overriden = p.override_settings.get("yandere_inpaint_resolution", None)
        if overriden:
            return overriden
    res = shared.opts.data.get("yandere_inpaint_resolution", 512)
    return res



yandere_upscaler_settings = {
    'yandere_inpaint_upscaler_name': shared.OptionInfo(
                ORIGINAL_NAME,
                "Override yandere inpaint model name",
                gr.Textbox,
            ),

    'yandere_inpaint_tile_size': shared.OptionInfo(
                512,
                "Tile size for yandere inpainting",
                gr.Slider,
                {
                    "minimum": 128,
                    "maximum": 2048,
                    "step": 8,
                }
            ),

    'yandere_inpaint_upscaler': shared.OptionInfo(
            "ESRGAN_4x",
            "Upscaler for Yandere Inpaint masked content",
            gr.Dropdown,
            lambda: {"choices": [x.name for x in shared.sd_upscalers]},
        ).info("I recommend to use Waifu2x upscaler from extension, because it's very fast and good enough for this purpose"),

    'yandere_inpaint_resolution': shared.OptionInfo(
            320,
            "Resolution for Yandere Inpaint masked content",
            gr.Slider,
            {
                "minimum": 256,
                "maximum": 2048,
                "step": 8,
            },
        ).info("Reduce to avoid white filling"),
}

shared.options_templates.update(shared.options_section(('extras_inpaint', 'Extras Inpaint'), yandere_upscaler_settings))

