import logging
import os
import tempfile
import time

import gradio as gr
import numpy as np
import rembg
import torch
from PIL import Image
from functools import partial

from tsr.system import TSR
from tsr.utils import remove_background, resize_foreground, to_gradio_3d_orientation

import argparse


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

# Load TripoSR model from local directory
model_path = os.path.join(os.path.dirname(__file__), "models", "stabilityai--TripoSR")
model = TSR.from_pretrained(
    model_path,
    config_name="config.yaml",
    weight_name="model.ckpt",
)

# chunk size will be set dynamically based on user input
model.to(device)

rembg_session = rembg.new_session()


def check_input_image(input_image):
    if input_image is None:
        raise gr.Error("No image uploaded!")


def preprocess(input_image, do_remove_background, foreground_ratio):
    def fill_background(image):
        image = np.array(image).astype(np.float32) / 255.0
        image = image[:, :, :3] * image[:, :, 3:4] + (1 - image[:, :, 3:4]) * 0.5
        image = Image.fromarray((image * 255.0).astype(np.uint8))
        return image

    if do_remove_background:
        image = input_image.convert("RGB")
        image = remove_background(image, rembg_session)
        image = resize_foreground(image, foreground_ratio)
        image = fill_background(image)
    else:
        image = input_image
        if image.mode == "RGBA":
            image = fill_background(image)
    return image


def generate(image, mc_resolution, chunk_size, density_threshold, decimation_target, 
             smooth_iterations, smooth_lambda, remesh_size, formats=["obj", "glb"]):
    # Set chunk size
    model.renderer.set_chunk_size(chunk_size)
    
    scene_codes = model(image, device=device)
    mesh = model.extract_mesh(
        scene_codes, 
        True, 
        resolution=mc_resolution,
        threshold=density_threshold,
        decimation_target=decimation_target,
        smooth_iterations=smooth_iterations,
        smooth_lambda=smooth_lambda,
        remesh_size=remesh_size
    )[0]
    mesh = to_gradio_3d_orientation(mesh)
    rv = []
    for format in formats:
        mesh_path = tempfile.NamedTemporaryFile(suffix=f".{format}", delete=False)
        mesh.export(mesh_path.name)
        rv.append(mesh_path.name)
    
    # Return OBJ download, GLB for OBJ tab viewer, GLB for GLB tab
    return rv[0], rv[1], rv[1]


def run_example(image_pil):
    preprocessed = preprocess(image_pil, False, 0.9)
    mesh_name_obj_download, mesh_name_obj_view, mesh_name_glb = generate(
        preprocessed, 
        256,  # mc_resolution
        8192,  # chunk_size
        25,  # density_threshold
        10000,  # decimation_target
        3,  # smooth_iterations
        0.5,  # smooth_lambda
        0.01,  # remesh_size
        ["obj", "glb"]
    )
    return preprocessed, mesh_name_obj_download, mesh_name_obj_view, mesh_name_glb


with gr.Blocks(title="TripoSR") as interface:
    gr.Markdown(
        """
    # TripoSR Premium v1 by SECourses : https://www.patreon.com/posts/141896910
    """
    )
    with gr.Row(variant="panel"):
        with gr.Column():
            with gr.Row():
                input_image = gr.Image(
                    label="Input Image",
                    image_mode="RGBA",
                    sources="upload",
                    type="pil",
                    elem_id="content_image",
                )
                processed_image = gr.Image(label="Processed Image", interactive=False)
            with gr.Row():
                with gr.Group():
                    do_remove_background = gr.Checkbox(
                        label="Remove Background", value=True
                    )
                    foreground_ratio = gr.Slider(
                        label="Foreground Ratio",
                        minimum=0.5,
                        maximum=1.0,
                        value=0.85,
                        step=0.05,
                    )
                    mc_resolution = gr.Slider(
                        label="Marching Cubes Resolution",
                        minimum=32,
                        maximum=512,
                        value=320,
                        step=32
                    )
                    chunk_size = gr.Slider(
                        label="Chunk Size (VRAM vs Speed)",
                        minimum=2048,
                        maximum=16384,
                        value=8192,
                        step=2048
                    )
                    density_threshold = gr.Slider(
                        label="Density Threshold",
                        minimum=10,
                        maximum=50,
                        value=25,
                        step=1
                    )
                    decimation_target = gr.Slider(
                        label="Decimation Target (faces)",
                        minimum=1000,
                        maximum=50000,
                        value=10000,
                        step=1000
                    )
                    smooth_iterations = gr.Slider(
                        label="Smoothing Iterations",
                        minimum=0,
                        maximum=10,
                        value=3,
                        step=1
                    )
                    smooth_lambda = gr.Slider(
                        label="Smoothing Lambda",
                        minimum=0.1,
                        maximum=1.0,
                        value=0.5,
                        step=0.1
                    )
                    remesh_size = gr.Slider(
                        label="Remesh Size",
                        minimum=0.005,
                        maximum=0.05,
                        value=0.01,
                        step=0.005
                    )
            with gr.Row():
                submit = gr.Button("Generate", elem_id="generate", variant="primary")
        with gr.Column():
            with gr.Row():
                with gr.Tab("GLB"):
                    output_model_glb = gr.Model3D(
                        label="Output Model (GLB Format)",
                        interactive=False,
                    )
                    gr.Markdown("Note: The model shown here has a darker appearance. Download to get correct results.")            
                with gr.Tab("OBJ"):
                    output_model_obj_view = gr.Model3D(
                        label="Model Preview (using GLB)",
                        interactive=False,
                    )
                    output_model_obj_download = gr.File(
                        label="Download OBJ File",
                        interactive=False,
                    )
                    gr.Markdown("Note: Preview shows GLB format. Download button provides OBJ file.")
            with gr.Row():
                gr.Markdown(
            """
            ### Tips:
            1. If you find the result is unsatisfied, please try to change the foreground ratio. It might improve the results.
            2. It's better to disable "Remove Background" for the provided examples (except for the last one) since they have been already preprocessed.
            3. Otherwise, please disable "Remove Background" option only if your input image is RGBA with transparent background, image contents are centered and occupy more than 70% of image width or height.
            
            ### Advanced Parameters:
            - **Marching Cubes Resolution** (32-512): Higher = more detail but slower and uses more VRAM
            - **Chunk Size**: Lower = less VRAM usage but slower processing
            - **Density Threshold** (10-50): Lower = captures more detail, Higher = smoother mesh
            - **Decimation Target**: Controls final mesh complexity (faces count)
            - **Smoothing**: Adjust iterations and lambda for mesh smoothness
            - **Remesh Size**: Controls triangle size in mesh cleaning
            """
        )
    with gr.Row(variant="panel"):
        gr.Examples(
            examples=[
                "examples/hamburger.png",
                "examples/poly_fox.png",
                "examples/robot.png",
                "examples/teapot.png",
                "examples/tiger_girl.png",
                "examples/horse.png",
                "examples/flamingo.png",
                "examples/unicorn.png",
                "examples/chair.png",
                "examples/iso_house.png",
                "examples/marble.png",
                "examples/police_woman.png",
                "examples/captured.jpeg",
            ],
            inputs=[input_image],
            outputs=[processed_image, output_model_obj_download, output_model_obj_view, output_model_glb],
            cache_examples=False,
            fn=partial(run_example),
            label="Examples",
            examples_per_page=20,
        )
    submit.click(fn=check_input_image, inputs=[input_image]).success(
        fn=preprocess,
        inputs=[input_image, do_remove_background, foreground_ratio],
        outputs=[processed_image],
    ).success(
        fn=generate,
        inputs=[processed_image, mc_resolution, chunk_size, density_threshold, decimation_target,
                smooth_iterations, smooth_lambda, remesh_size],
        outputs=[output_model_obj_download, output_model_obj_view, output_model_glb],
    )



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--username', type=str, default=None, help='Username for authentication')
    parser.add_argument('--password', type=str, default=None, help='Password for authentication')
    parser.add_argument('--port', type=int, default=7860, help='Port to run the server listener on')
    parser.add_argument("--listen", action='store_true', help="launch gradio with 0.0.0.0 as server name, allowing to respond to network requests")
    parser.add_argument("--share", action='store_true', help="use share=True for gradio and make the UI accessible through their site")
    parser.add_argument("--queuesize", type=int, default=1, help="launch gradio queue max_size")
    args = parser.parse_args()
    interface.queue(max_size=args.queuesize)
    interface.launch(
        auth=(args.username, args.password) if (args.username and args.password) else None,
        share=args.share,inbrowser=True
    )