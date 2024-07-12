from modules import shared
from modules.processing import StableDiffusionProcessingImg2Img
import gradio as gr



def getPreprocessorResolution(p: StableDiffusionProcessingImg2Img = None):
    if hasattr(p, 'override_settings'):
        overriden = p.override_settings.get("manga_inpainting_preprocessor_resolution", None)
        if overriden:
            return overriden
    res = shared.opts.data.get("manga_inpainting_preprocessor_resolution", 512)
    return res



manga_inpainting_settings = {
    'manga_inpainting_preprocessor_resolution': shared.OptionInfo(
            512,
            "Resolution for CN lineart_anime_denoise preprocessor which is used inside Manga Inpainting",
            gr.Slider,
            {
                "minimum": 256,
                "maximum": 2048,
                "step": 8,
            },
        ),
}

shared.options_templates.update(shared.options_section(('extras_inpaint', 'Extras Inpaint'), manga_inpainting_settings))

