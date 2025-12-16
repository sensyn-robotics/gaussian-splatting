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

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import numpy as np
import argparse

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False


def render_set(model_path, name, iteration, views, gaussians, pipeline, background, train_test_exp, separate_sh):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "depth")
    depth_vis_path = os.path.join(model_path, name, "ours_{}".format(iteration), "depth_vis")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)
    makedirs(depth_vis_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        result = render(view, gaussians, pipeline, background, use_trained_exp=train_test_exp, separate_sh=separate_sh)
        rendering = result["render"]
        depth = result["depth"]
        gt = view.original_image[0:3, :, :]

        if args.train_test_exp:
            rendering = rendering[..., rendering.shape[-1] // 2:]
            depth = depth[..., depth.shape[-1] // 2:]
            gt = gt[..., gt.shape[-1] // 2:]

        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

        # Save Raw Depth as .npy
        np.save(os.path.join(depth_path, '{0:05d}'.format(idx) + ".npy"), depth.cpu().numpy())

        # Save Visualization (Normalize for visibility, simple gray scale)
        depth_vis = depth.clone().detach()
        depth_vis = (depth_vis - depth_vis.min()) / (depth_vis.max() - depth_vis.min() + 1e-8)
        torchvision.utils.save_image(depth_vis, os.path.join(depth_vis_path, '{0:05d}'.format(idx) + ".png"))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, separate_sh: bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, dataset.train_test_exp, separate_sh)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, dataset.train_test_exp, separate_sh)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    
    # Custom argument parsing to handle missing cfg_args (common in some checkpoints)
    import sys
    args = parser.parse_args(sys.argv[1:])
    
    # Try to load cfg_args if available, but don't crash
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
        # Ensure source_path is absolute if provided
        if hasattr(args, "source_path") and args.source_path:
             args.source_path = os.path.abspath(args.source_path)
        
        # Manually set defaults for required params if they are None (due to sentinel=True)
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

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, SPARSE_ADAM_AVAILABLE)
