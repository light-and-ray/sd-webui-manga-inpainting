# Manga Inpainting

This is an extension for [stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui) which adds [msxie92/MangaInpainting](https://github.com/msxie92/MangaInpainting) inside extras tab. Requires [lama-cleaner-masked-content](https://github.com/light-and-ray/sd-webui-lama-cleaner-masked-content) because this extensions compilates all my extra inpainting methods inside extras tab

![](/images/comparasion.png)

*Comparison with lama (right)*

![](/images/preview.jpg)

Installation:
1. Install [sd-webui-controlnet](https://github.com/Mikubill/sd-webui-controlnet) (required for `lineart_anime_denoise` preprocessor which is used in manga inpainting algorithm, and required by lama)
2. Install [lama-cleaner-masked-content](https://github.com/light-and-ray/sd-webui-lama-cleaner-masked-content)
3. Install this extension
4. Done!