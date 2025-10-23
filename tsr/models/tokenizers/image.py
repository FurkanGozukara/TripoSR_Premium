from dataclasses import dataclass
import json
import os

import torch
import torch.nn as nn
from einops import rearrange
from transformers.models.vit.modeling_vit import ViTModel

from ...utils import BaseModule


class DINOSingleImageTokenizer(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        pretrained_model_name_or_path: str = "facebook/dino-vitb16"
        enable_gradient_checkpointing: bool = False

    cfg: Config

    def configure(self) -> None:
        # Handle special cases for local DINO model
        if self.cfg.pretrained_model_name_or_path == "__LOCAL_DINO_MODEL__" or self.cfg.pretrained_model_name_or_path == "facebook/dino-vitb16":
            # Resolve to the local DINO model path relative to this module
            module_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))  # TripoSR_Premium
            self.cfg.pretrained_model_name_or_path = os.path.join(module_dir, "models", "facebook--dino-vitb16")

        # Check if it's a local path (contains separator or is a relative/absolute path)
        is_local_path = (
            os.sep in self.cfg.pretrained_model_name_or_path or
            (os.altsep and os.altsep in self.cfg.pretrained_model_name_or_path) or
            self.cfg.pretrained_model_name_or_path.startswith('.') or
            self.cfg.pretrained_model_name_or_path.startswith('..') or
            os.path.isabs(self.cfg.pretrained_model_name_or_path)
        )

        if is_local_path:
            # Local path - load directly
            config_path = os.path.join(self.cfg.pretrained_model_name_or_path, "config.json")
            model_path = os.path.join(self.cfg.pretrained_model_name_or_path, "pytorch_model.bin")

            # Load config from local file
            with open(config_path, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)

            self.model: ViTModel = ViTModel(
                ViTModel.config_class(**config_dict)
            )

            # Load model weights if local path
            if os.path.exists(model_path):
                state_dict = torch.load(model_path, map_location="cpu")
                # Filter out missing keys (like pooler) that might not be present
                missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
                if missing_keys:
                    print(f"Warning: Missing keys in DINO state_dict: {missing_keys}")
                if unexpected_keys:
                    print(f"Warning: Unexpected keys in DINO state_dict: {unexpected_keys}")
        else:
            # HF repo ID - use default
            from huggingface_hub import hf_hub_download
            config_path = hf_hub_download(
                repo_id=self.cfg.pretrained_model_name_or_path,
                filename="config.json",
            )

            self.model: ViTModel = ViTModel(
                ViTModel.config_class.from_pretrained(config_path)
            )

        if self.cfg.enable_gradient_checkpointing:
            self.model.encoder.gradient_checkpointing = True

        self.register_buffer(
            "image_mean",
            torch.as_tensor([0.485, 0.456, 0.406]).reshape(1, 1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "image_std",
            torch.as_tensor([0.229, 0.224, 0.225]).reshape(1, 1, 3, 1, 1),
            persistent=False,
        )

    def forward(self, images: torch.FloatTensor, **kwargs) -> torch.FloatTensor:
        packed = False
        if images.ndim == 4:
            packed = True
            images = images.unsqueeze(1)

        batch_size, n_input_views = images.shape[:2]
        images = (images - self.image_mean) / self.image_std
        out = self.model(
            rearrange(images, "B N C H W -> (B N) C H W"), interpolate_pos_encoding=True
        )
        local_features, global_features = out.last_hidden_state, out.pooler_output
        local_features = local_features.permute(0, 2, 1)
        local_features = rearrange(
            local_features, "(B N) Ct Nt -> B N Ct Nt", B=batch_size
        )
        if packed:
            local_features = local_features.squeeze(1)

        return local_features

    def detokenize(self, *args, **kwargs):
        raise NotImplementedError
