# Face Anonymization

This repo contains a simple image anonymization suite. It supports blurring faces, inpainting new faces using diffusion, and blurring license plates.




## How to use
- ``` pip install -r requirements.txt```
- to blur faces: ```python anonymize.py blur <path to your image>```
- to replaces faces: ```python anonymize.py inpaint <path to your image>```


## Output
### Faces
Original Image:
![Original](examples/friends.jpg)
Blurred Anonimization:
![Blurred](examples/friends_blur.jpg)
Inpainting Anonimization:
![Inpainted](examples/friends_inpaint.jpg)
