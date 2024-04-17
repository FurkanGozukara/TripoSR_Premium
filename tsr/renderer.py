import moderngl
import numpy as np
import os
import torch
import trimesh
import xatlas
from PIL import Image


class Renderer(object):

    def __init__(self, tsr, scene_codes, mesh, texture_resolution=1024, texture_padding=10):
        self.tsr = tsr
        self.scene_codes = scene_codes
        self.mesh = mesh
        self.texture_resolution = texture_resolution
        self.texture_padding = texture_padding

    def render(self):
        xatlas_result = self._run_xatlas()

        rasterize_result = self._run_rasterize(xatlas_result)

        tsr_result = {
            'model': self.tsr,
            'scene_codes': self.scene_codes,
        }

        bake_result = self._run_bake(tsr_result, rasterize_result)

        baked_img = Image.fromarray(
            (
                    bake_result.reshape(self.texture_resolution, self.texture_resolution, 4)
                    * 255.0
            ).astype(np.uint8)
        ).transpose(Image.FLIP_TOP_BOTTOM)

        # Create a PBRMaterial object with the baked texture image
        material = trimesh.visual.material.PBRMaterial(
            baseColorTexture=baked_img,
            metallicFactor=0.0,
            roughnessFactor=1.0,
        )

        # Create a TextureVisuals object with the material
        texture_visuals = trimesh.visual.TextureVisuals(
            uv=xatlas_result["uvs"],
            material=material
        )

        # Create the textured mesh with the updated visuals
        textured_mesh = trimesh.Trimesh(
            vertices=self.mesh.vertices[xatlas_result["vmapping"]],
            faces=xatlas_result["indices"],
            visual=texture_visuals
        )

        return textured_mesh

    def _run_xatlas(self):
        atlas = xatlas.Atlas()
        atlas.add_mesh(self.mesh.vertices, self.mesh.faces)
        options = xatlas.PackOptions()
        options.resolution = self.texture_resolution
        options.padding = self.texture_padding
        options.blockAlign = True
        options.bilinear = False
        options.bruteForce = True
        atlas.generate(pack_options=options)
        vmapping, indices, uvs = atlas[0]
        return {
            "vmapping": vmapping,
            "indices": indices,
            "uvs": uvs,
        }

    def _run_rasterize(self, xatlas_result):
        if os.name == 'nt':
            ctx = moderngl.create_context(standalone=True)
        else:
            ctx = moderngl.create_context(standalone=True, backend='egl')
        basic_prog = ctx.program(
            vertex_shader="""
                #version 330
                in vec2 in_uv;
                in vec3 in_pos;
                out vec3 v_pos;
                void main() {
                    v_pos = in_pos;
                    gl_Position = vec4(in_uv * 2.0 - 1.0, 0.0, 1.0);
                }
            """,
            fragment_shader="""
                #version 330
                in vec3 v_pos;
                out vec4 o_col;
                void main() {
                    o_col = vec4(v_pos, 1.0);
                }
            """,
        )
        gs_prog = ctx.program(
            vertex_shader="""
                #version 330
                in vec2 in_uv;
                in vec3 in_pos;
                out vec3 vg_pos;
                void main() {
                    vg_pos = in_pos;
                    gl_Position = vec4(in_uv * 2.0 - 1.0, 0.0, 1.0);
                }
            """,
            geometry_shader="""
                #version 330
                uniform float u_resolution;
                uniform float u_dilation;
                layout (triangles) in;
                layout (triangle_strip, max_vertices = 12) out;
                in vec3 vg_pos[];
                out vec3 vf_pos;
                void lineSegment(int aidx, int bidx) {
                    vec2 a = gl_in[aidx].gl_Position.xy;
                    vec2 b = gl_in[bidx].gl_Position.xy;
                    vec3 aCol = vg_pos[aidx];
                    vec3 bCol = vg_pos[bidx];
    
                    vec2 dir = normalize((b - a) * u_resolution);
                    vec2 offset = vec2(-dir.y, dir.x) * u_dilation / u_resolution;
    
                    gl_Position = vec4(a + offset, 0.0, 1.0);
                    vf_pos = aCol;
                    EmitVertex();
                    gl_Position = vec4(a - offset, 0.0, 1.0);
                    vf_pos = aCol;
                    EmitVertex();
                    gl_Position = vec4(b + offset, 0.0, 1.0);
                    vf_pos = bCol;
                    EmitVertex();
                    gl_Position = vec4(b - offset, 0.0, 1.0);
                    vf_pos = bCol;
                    EmitVertex();
                }
                void main() {
                    lineSegment(0, 1);
                    lineSegment(1, 2);
                    lineSegment(2, 0);
                    EndPrimitive();
                }
            """,
            fragment_shader="""
                #version 330
                in vec3 vf_pos;
                out vec4 o_col;
                void main() {
                    o_col = vec4(vf_pos, 1.0);
                }
            """,
        )
        uvs = xatlas_result["uvs"].flatten().astype("f4")
        pos = self.mesh.vertices[xatlas_result["vmapping"]].flatten().astype("f4")
        indices = xatlas_result["indices"].flatten().astype("i4")
        vbo_uvs = ctx.buffer(uvs)
        vbo_pos = ctx.buffer(pos)
        ibo = ctx.buffer(indices)
        vao_content = [
            vbo_uvs.bind("in_uv", layout="2f"),
            vbo_pos.bind("in_pos", layout="3f"),
        ]
        basic_vao = ctx.vertex_array(basic_prog, vao_content, ibo)
        gs_vao = ctx.vertex_array(gs_prog, vao_content, ibo)

        texture = ctx.texture(
            (self.texture_resolution, self.texture_resolution), 4, dtype='f4')
        texture.use(location=0)
        texture.build_mipmaps()
        texture.anisotropic = 16.0

        texture.min_filter = 'linear_mipmap_linear'
        texture.mag_filter = 'linear'

        fbo = ctx.framebuffer(
            color_attachments=[
                ctx.texture(
                    (self.texture_resolution, self.texture_resolution), 4, dtype="f4"
                )
            ]
        )
        fbo.use()
        fbo.clear(0.0, 0.0, 0.0, 0.0)
        gs_prog["u_resolution"].value = self.texture_resolution
        gs_prog["u_dilation"].value = self.texture_padding
        gs_vao.render()
        basic_vao.render()

        fbo_bytes = fbo.color_attachments[0].read()
        fbo_np = np.frombuffer(fbo_bytes, dtype="f4").reshape(
            self.texture_resolution, self.texture_resolution, 4
        )
        return fbo_np

    def _run_bake(self, tsr_result, rasterize_result):
        positions = torch.tensor(rasterize_result.reshape(-1, 4)[:, :-1]).to(tsr_result["scene_codes"].device)
        with torch.no_grad():
            queried_grid = tsr_result["model"].renderer.query_triplane(
                tsr_result["model"].decoder,
                positions,
                tsr_result["scene_codes"][0],
            )
        rgb_f = queried_grid["color"].cpu().numpy().reshape(-1, 3)
        rgba_f = np.insert(rgb_f, 3, rasterize_result.reshape(-1, 4)[:, -1], axis=1)
        rgba_f[rgba_f[:, -1] == 0.0] = [0, 0, 0, 0]
        return rgba_f.reshape(self.texture_resolution, self.texture_resolution, 4)
