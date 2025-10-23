from math import radians

import PIL.Image
import numpy as np
import os
import torch
import trimesh
from PIL import Image
from dataclasses import dataclass
from einops import rearrange
from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf
from typing import List, Union

from tsr.renderer import Renderer
from .mesh_utils import decimate_mesh, clean_mesh, laplacian_smooth
from .models.isosurface import MarchingCubeHelper
from .utils import (
    BaseModule,
    ImagePreprocessor,
    find_class,
    get_spherical_cameras,
    scale_tensor,
)


class TSR(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        cond_image_size: int

        image_tokenizer_cls: str
        image_tokenizer: dict

        tokenizer_cls: str
        tokenizer: dict

        backbone_cls: str
        backbone: dict

        post_processor_cls: str
        post_processor: dict

        decoder_cls: str
        decoder: dict

        renderer_cls: str
        renderer: dict

    cfg: Config

    @classmethod
    def from_pretrained(
            cls, pretrained_model_name_or_path: str, config_name: str, weight_name: str
    ):
        if os.path.isdir(pretrained_model_name_or_path):
            config_path = os.path.join(pretrained_model_name_or_path, config_name)
            weight_path = os.path.join(pretrained_model_name_or_path, weight_name)
        else:
            config_path = hf_hub_download(
                repo_id=pretrained_model_name_or_path, filename=config_name
            )
            weight_path = hf_hub_download(
                repo_id=pretrained_model_name_or_path, filename=weight_name
            )

        cfg = OmegaConf.load(config_path)
        OmegaConf.resolve(cfg)
        model = cls(cfg)
        ckpt = torch.load(weight_path, map_location="cpu")
        model.load_state_dict(ckpt)
        return model

    def configure(self):
        self.image_tokenizer = find_class(self.cfg.image_tokenizer_cls)(
            self.cfg.image_tokenizer
        )
        self.tokenizer = find_class(self.cfg.tokenizer_cls)(self.cfg.tokenizer)
        self.backbone = find_class(self.cfg.backbone_cls)(self.cfg.backbone)
        self.post_processor = find_class(self.cfg.post_processor_cls)(
            self.cfg.post_processor
        )
        self.decoder = find_class(self.cfg.decoder_cls)(self.cfg.decoder)
        self.renderer = find_class(self.cfg.renderer_cls)(self.cfg.renderer)
        self.image_processor = ImagePreprocessor()
        self.isosurface_helper = None

    def forward(
            self,
            image: Union[
                PIL.Image.Image,
                np.ndarray,
                torch.FloatTensor,
                List[PIL.Image.Image],
                List[np.ndarray],
                List[torch.FloatTensor],
            ],
            device: str,
    ) -> torch.FloatTensor:
        rgb_cond = self.image_processor(image, self.cfg.cond_image_size)[:, None].to(
            device
        )
        batch_size = rgb_cond.shape[0]

        input_image_tokens: torch.Tensor = self.image_tokenizer(
            rearrange(rgb_cond, "B Nv H W C -> B Nv C H W", Nv=1),
        )

        input_image_tokens = rearrange(
            input_image_tokens, "B Nv C Nt -> B (Nv Nt) C", Nv=1
        )

        tokens: torch.Tensor = self.tokenizer(batch_size)

        tokens = self.backbone(
            tokens,
            encoder_hidden_states=input_image_tokens,
        )

        scene_codes = self.post_processor(self.tokenizer.detokenize(tokens))
        return scene_codes

    def render(
            self,
            scene_codes,
            n_views: int,
            elevation_deg: float = 0.0,
            camera_distance: float = 1.9,
            fovy_deg: float = 40.0,
            height: int = 256,
            width: int = 256,
            return_type: str = "pil",
    ):
        rays_o, rays_d = get_spherical_cameras(
            n_views, elevation_deg, camera_distance, fovy_deg, height, width
        )
        rays_o, rays_d = rays_o.to(scene_codes.device), rays_d.to(scene_codes.device)

        def process_output(image: torch.FloatTensor):
            if return_type == "pt":
                return image
            elif return_type == "np":
                return image.detach().cpu().numpy()
            elif return_type == "pil":
                return Image.fromarray(
                    (image.detach().cpu().numpy() * 255.0).astype(np.uint8)
                )
            else:
                raise NotImplementedError

        images = []
        for scene_code in scene_codes:
            images_ = []
            for i in range(n_views):
                with torch.no_grad():
                    image = self.renderer(
                        self.decoder, scene_code, rays_o[i], rays_d[i]
                    )
                images_.append(process_output(image))
            images.append(images_)

        return images

    def set_marching_cubes_resolution(self, resolution: int):
        if (
                self.isosurface_helper is not None
                and self.isosurface_helper.resolution == resolution
        ):
            return
        self.isosurface_helper = MarchingCubeHelper(resolution)

    def extract_mesh(self, scene_codes, has_vertex_color=True, resolution: int = 256, threshold: float = 25.0,
                     remesh_size: float = 0.01, decimation_target: int = 10000, 
                     smooth_iterations: int = 3, smooth_lambda: float = 0.5):
        self.set_marching_cubes_resolution(resolution)
        meshes = []
        for scene_code in scene_codes:
            with torch.no_grad():
                density = self.renderer.query_triplane(
                    self.decoder,
                    scale_tensor(
                        self.isosurface_helper.grid_vertices.to(scene_codes.device),
                        self.isosurface_helper.points_range,
                        (-self.renderer.cfg.radius, self.renderer.cfg.radius),
                    ),
                    scene_code,
                )["density_act"]
            v_pos, t_pos_idx = self.isosurface_helper(-(density - threshold))
            v_pos = scale_tensor(
                v_pos,
                self.isosurface_helper.points_range,
                (-self.renderer.cfg.radius, self.renderer.cfg.radius),
            )

            verts, faces = v_pos.cpu().numpy(), t_pos_idx.cpu().numpy()

            verts, faces = clean_mesh(verts, faces, remesh=True, remesh_size=remesh_size)
            verts, faces = decimate_mesh(verts, faces, target=decimation_target)
            smooth_verts = laplacian_smooth(verts, faces, num_iterations=smooth_iterations, lambda_factor=smooth_lambda)

            # Query the renderer to get vertex colors for the smoothed mesh
            with torch.no_grad():
                smooth_verts_tensor = torch.from_numpy(smooth_verts).float().to(scene_codes.device)
                smooth_colors = self.renderer.query_triplane(
                    self.decoder,
                    smooth_verts_tensor,
                    scene_code,
                )["color"]
                smooth_vcolors = smooth_colors.cpu().numpy()

            tmp_mesh = trimesh.Trimesh(vertices=smooth_verts, faces=faces, vertex_colors=smooth_vcolors)

            renderer = Renderer(self, scene_codes, tmp_mesh)
            textured_mesh = renderer.render()

            rotation_x = 90
            rotation_y = 255
            rotation_z = 180

            # Convert the angles from degrees to radians
            rotation_x = radians(rotation_x)
            rotation_y = radians(rotation_y)
            rotation_z = radians(rotation_z)

            # Create the rotation matrices for each axis
            rotation_matrix_x = trimesh.transformations.rotation_matrix(rotation_x, [1, 0, 0])
            rotation_matrix_y = trimesh.transformations.rotation_matrix(rotation_y, [0, 1, 0])
            rotation_matrix_z = trimesh.transformations.rotation_matrix(rotation_z, [0, 0, 1])

            # Combine the rotation matrices by multiplying them together
            rotation_matrix = np.dot(rotation_matrix_z, np.dot(rotation_matrix_y, rotation_matrix_x))

            # Apply the rotation to the mesh
            textured_mesh.apply_transform(rotation_matrix)

            # This process allows us to do shade smoothing
            vertex_normals = textured_mesh.vertex_normals

            # Create a new mesh with the vertex normals
            textured_mesh = trimesh.Trimesh(
                vertices=textured_mesh.vertices,
                faces=textured_mesh.faces,
                vertex_normals=vertex_normals,
                visual=textured_mesh.visual
            )

            meshes.append(textured_mesh)
        return meshes
