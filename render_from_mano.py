import torch
from torch.utils.data import DataLoader
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
import concurrent.futures
import multiprocessing
from pathlib import Path
from PIL import Image
import numpy as np
import json
import glob

from gaussian_renderer import render
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import ManoGaussianModel
from mesh_renderer import NVDiffRenderer


mesh_renderer = NVDiffRenderer()

def write_data(path2data):
    for path, data in path2data.items():
        if not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)

        if path.suffix in [".png", ".jpg"]:
            data = data.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
            Image.fromarray(data).save(path)
        elif path.suffix in [".obj"]:
            with open(path, "w") as f:
                f.write(data)
        elif path.suffix in [".txt"]:
            with open(path, "w") as f:
                f.write(data)
        elif path.suffix in [".npz"]:
            np.savez(path, **data)
        else:
            raise NotImplementedError(f"Unknown file type: {path.suffix}")

def load_single_mano_param(json_path, shape=None):
    """
    단일 JSON 파일에서 MANO 파라미터를 읽어옵니다.
    
    Args:
        json_path: MANO 파라미터 JSON 파일 경로
        shape: shape 파라미터 (없으면 JSON에서 읽음)
        static_offset: static_offset 파라미터 (없으면 0으로 초기화)
        
    Returns:
        mano_param 딕셔너리: {
            'shape': torch.Tensor,
            'root_pose': torch.Tensor,  # (1, 3)
            'root_trans': torch.Tensor,  # (1, 3)
            'hand_pose': torch.Tensor,  # (1, 45)
        }
    """
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"MANO parameter file not found: {json_path}")
    
    with open(json_path, 'r') as f:
        param = json.load(f)
    
    # shape는 모든 프레임에서 동일하므로 인자로 받거나 JSON에서 읽음
    if shape is None:
        shape = torch.from_numpy(np.array(param['shape'])).float().cuda()
    else:
        shape = shape  # 이미 CUDA에 있음
    
    # root_pose, root_trans, hand_pose를 읽어서 (1, ...) 형태로 변환
    root_pose = torch.from_numpy(np.array(param['root_pose'])).float()
    if root_pose.dim() == 1:
        root_pose = root_pose.unsqueeze(0)  # (1, 3)
    
    root_trans = torch.from_numpy(np.array(param['root_trans'])).float()
    if root_trans.dim() == 2:
        root_trans = root_trans.squeeze(0)  # (3,)
    if root_trans.dim() == 1:
        root_trans = root_trans.unsqueeze(0)  # (1, 3)
    
    hand_pose = torch.from_numpy(np.array(param['hand_pose'])).float()
    if hand_pose.dim() == 1:
        hand_pose = hand_pose.unsqueeze(0)  # (1, 45)
    
    mano_param = {
        'shape': shape,
        'root_pose': root_pose.cuda(),
        'root_trans': root_trans.cuda(),
        'hand_pose': hand_pose.cuda(),
    }
    
    return mano_param


        
if __name__ == "__main__":
    ### Usage
    # python render_from_mano.py -m output/custom_2_no_loss_masking/ --mano_params_folder data/subject_1_merged1358/mano_params/params/
    # Set up command line argument parser
    parser = ArgumentParser(description="Render from checkpoint with MANO parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--mano_params_folder", type=str, default=None,
                       help="Path to mano_params/params/ folder containing JSON files")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)
    
    if args.mano_params_folder:
        print(f"MANO parameters folder: {args.mano_params_folder}")

    # Initialize system state (RNG)
    safe_state(args.quiet)

    dataset = model.extract(args)
    pipeline = pipeline.extract(args)

    gaussians = ManoGaussianModel(dataset.sh_degree)

    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    views = scene.getTrainCameras()
    view = views[0]

    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # Set output path
    iter_path = Path(dataset.model_path) / "mano_renders" / f"ours_{args.iteration}"
    render_path = iter_path / "renders"
    # gts_path = iter_path / "gt"
    makedirs(render_path, exist_ok=True)
    # makedirs(gts_path, exist_ok=True)


    # 체크포인트에서 로드한 학습된 shape와 static_offset 사용
    if gaussians.mano_param is None:
        raise ValueError("MANO parameters not loaded from checkpoint. Make sure the checkpoint contains mano_param.npz")
    
    shape = gaussians.mano_param['shape']  # 학습된 shape 사용
    print('Loaded shape from checkpoint: ', shape.shape)

    # Load MANO parameter files
    mano_params_list = sorted(glob.glob(args.mano_params_folder + "/*.json"))
    print(f"Found {len(mano_params_list)} MANO parameter files.")

    for i, mano_param_path in enumerate(mano_params_list):
        mano_param = load_single_mano_param(mano_param_path, shape=shape)
        gaussians.update_mesh_by_param_dict(mano_param)

        rendering = render(view, gaussians, pipeline, background)["render"]
        # gt = view.original_image[0:3, :, :]

        write_data({Path(render_path) / f'{i:05d}.png': rendering})



