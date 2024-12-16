from .autoencoder_magvit import AutoencoderKLCogVideoX, AutoencoderKLMagvit, AutoencoderKL
from .transformer3d import (
    EasyAnimateTransformer3DModel,
    HunyuanTransformer3DModel,
    Transformer3DModel,
    EasyAnimateTransformer3DModelCameraControlV1,
    EasyAnimateTransformer3DModelCameraControlV2,
)


name_to_transformer3d = {
    "Transformer3DModel": Transformer3DModel,
    "HunyuanTransformer3DModel": HunyuanTransformer3DModel,
    "EasyAnimateTransformer3DModel": EasyAnimateTransformer3DModel,
    "EasyAnimateTransformer3DModelCameraControlV1": EasyAnimateTransformer3DModelCameraControlV1,
    "EasyAnimateTransformer3DModelCameraControlV2": EasyAnimateTransformer3DModelCameraControlV2,
}
name_to_autoencoder_magvit = {
    "AutoencoderKL": AutoencoderKL,
    "AutoencoderKLMagvit": AutoencoderKLMagvit,
    "AutoencoderKLCogVideoX": AutoencoderKLCogVideoX,
}
