#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import argparse
from argparse import ArgumentParser
import os
import sys
import numpy as np
import torch
import torchvision
from tqdm import tqdm

from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel, render
from scene import Scene
from utils.general_utils import safe_state

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False


def render_set(model_path, name, iteration, views, gaussians, pipeline, background, train_test_exp, separate_sh, output_path=None):
    if output_path:
        # If explicit output path is given, everything goes there under train/test
        base_path = os.path.join(output_path, name)
    else:
        # Legacy behavior
        base_path = os.path.join(model_path, name, "ours_{}".format(iteration))

    render_path = os.path.join(base_path, "renders")
    gts_path = os.path.join(base_path, "gt")
    depth_path = os.path.join(base_path, "depth")
    depth_vis_path = os.path.join(base_path, "depth_vis")

    os.makedirs(render_path, exist_ok=True)
    os.makedirs(gts_path, exist_ok=True)
    os.makedirs(depth_path, exist_ok=True)
    os.makedirs(depth_vis_path, exist_ok=True)

    # Sort views by image_name to ensure consistent ordering
    sorted_views = sorted(views, key=lambda v: v.image_name)

    for idx, view in enumerate(tqdm(sorted_views, desc="Rendering progress")):
        result = render(view, gaussians, pipeline, background, use_trained_exp=train_test_exp, separate_sh=separate_sh)
        rendering = result["render"]
        depth = result["depth"]
        gt = view.original_image[0:3, :, :]

        if args.train_test_exp:
            rendering = rendering[..., rendering.shape[-1] // 2:]
            depth = depth[..., depth.shape[-1] // 2:]
            gt = gt[..., gt.shape[-1] // 2:]

        # Use original image name (without extension) for consistent correspondence
        image_name = os.path.splitext(view.image_name)[0]

        torchvision.utils.save_image(rendering, os.path.join(render_path, image_name + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, image_name + ".png"))

        # Save Raw Depth as .npy
        np.save(os.path.join(depth_path, image_name + ".npy"), depth.cpu().numpy())

        # Save Visualization (Normalize for visibility, simple gray scale)
        depth_vis = depth.clone().detach()
        depth_vis = (depth_vis - depth_vis.min()) / (depth_vis.max() - depth_vis.min() + 1e-8)
        torchvision.utils.save_image(depth_vis, os.path.join(depth_vis_path, image_name + ".png"))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, separate_sh: bool, ply_path: str = None, output_path: str = None):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, ply_path=ply_path)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, dataset.train_test_exp, separate_sh, output_path=output_path)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, dataset.train_test_exp, separate_sh, output_path=output_path)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="""Render depth maps from a trained Gaussian Splatting model.

Required parameters:
  --source_path, -s    Path to the dataset (COLMAP scene with cameras.bin/txt)
  --model_path, -m     Path to the trained model directory (contains cfg_args and point_cloud/)

Usage modes:
  1. Standard mode (from trained model):
     python render_depth.py -m <model_path> -s <source_path>

  2. Explicit PLY mode (custom ply file):
     python render_depth.py -s <source_path> --ply_file <path/to/point_cloud.ply> --output_path <output_dir>

Examples:
  python render_depth.py -m output/truck -s data/truck
  python render_depth.py -s data/truck --ply_file custom.ply --output_path renders/
""", formatter_class=argparse.RawDescriptionHelpFormatter)
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int, help="Iteration to load (default: -1 for latest)")
    parser.add_argument("--skip_train", action="store_true", help="Skip rendering train set")
    parser.add_argument("--skip_test", action="store_true", help="Skip rendering test set")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output")
    # New explicit arguments
    parser.add_argument("--ply_file", type=str, default=None, help="Explicit path to point_cloud.ply (use with --output_path)")
    parser.add_argument("--output_path", type=str, default=None, help="Explicit output directory (use with --ply_file)")
    
    args = parser.parse_args(sys.argv[1:])

    # Handle explicit PLY mode first - set model_path from output_path if not provided
    if args.ply_file and args.output_path and not args.model_path:
        args.model_path = args.output_path
        os.makedirs(args.model_path, exist_ok=True)

    # Try to load cfg_args if available, but don't crash
    if args.model_path:
        cfgfilepath = os.path.join(args.model_path, "cfg_args")
        if os.path.exists(cfgfilepath):
            print("Looking for config file in", cfgfilepath)
            with open(cfgfilepath) as cfg_file:
                print("Config file found: {}".format(cfgfilepath))
                cfgfile_string = cfg_file.read()
                args_cfgfile = eval(cfgfile_string)

                # Merge logic from get_combined_args
                merged_dict = vars(args_cfgfile).copy()
                for k,v in vars(args).items():
                    if v != None:
                        merged_dict[k] = v
                args = argparse.Namespace(**merged_dict)
        else:
            print(f"Config file not found at {cfgfilepath}, using command line arguments.")
    else:
        print("No model_path provided, using command line arguments.")

    # Ensure source_path is absolute if provided
    if hasattr(args, "source_path") and args.source_path:
         args.source_path = os.path.abspath(args.source_path)

    # Set defaults for required params if they are None (due to sentinel=True)
    if args.resolution is None:
        args.resolution = -1
    if args.sh_degree is None:
        args.sh_degree = 3
    if args.white_background is None:
        args.white_background = False
    if args.images is None:
        args.images = "images"
    if hasattr(args, "depths") and args.depths is None:
        args.depths = ""
    if args.data_device is None:
        args.data_device = "cuda"
    if args.eval is None:
        args.eval = False

    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Pass the explicit paths if they exist
    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, SPARSE_ADAM_AVAILABLE, ply_path=args.ply_file, output_path=args.output_path)
