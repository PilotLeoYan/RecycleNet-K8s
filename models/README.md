# Models Directory

The ONNX model is stored at Hugging Face:
[LeoYan/recyclenet-mobilenetv3](https://huggingface.co/LeoYan/recyclenet-mobilenetv3)

During startup, if models/recyclenet.onnx is not found locally, the API automatically downloads it from Hugging Face Hub using huggingface_hub.hf_hub_download.
