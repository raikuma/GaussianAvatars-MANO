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
import zmq, time
import cv2

from gaussian_renderer import render
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import ManoGaussianModel
from mesh_renderer import NVDiffRenderer
from mano_stream_protocol import unpack_mano


mesh_renderer = NVDiffRenderer()

def load_single_mano_param(payload, shape=None):
    if shape is None:
        shape = torch.from_numpy(np.array(payload["betas"])).float().cuda()
    else:
        shape = shape
    
    root_pose = torch.from_numpy(np.array(payload["global_orient"])).float()
    if root_pose.dim() == 1:
        root_pose = root_pose.unsqueeze(0)  # (1, 3)

    root_trans = torch.from_numpy(np.array(payload["transl"])).float()
    if root_trans.dim() == 2:
        root_trans = root_trans.squeeze(0)  # (3,)
    if root_trans.dim() == 1:
        root_trans = root_trans.unsqueeze(0)  # (1, 3)
        
    hand_pose = torch.from_numpy(np.array(payload["hand_pose"])).float()
    if hand_pose.dim() == 1:
        hand_pose = hand_pose.unsqueeze(0)  # (1, 45)

    return {
        'shape': shape,
        'root_pose': root_pose,
        'root_trans': root_trans,
        'hand_pose': hand_pose,
    }

def main(args, model, pipeline):
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.SUB)
    sock.setsockopt(zmq.SUBSCRIBE, b"mano")
    sock.setsockopt(zmq.RCVHWM, 10)     # 1도 가능, 디버그땐 10 추천
    sock.connect("tcp://127.0.0.1:5555")

    poller = zmq.Poller()
    poller.register(sock, zmq.POLLIN)

    print("[receiver] connected, waiting...", flush=True)

    # Initialize rendering pipeline
    dataset = model.extract(args)
    pipeline = pipeline.extract(args)

    gaussians = ManoGaussianModel(dataset.sh_degree)

    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    views = scene.getTrainCameras()

    view = views[0]
    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    shape = gaussians.mano_param['shape']  # 학습된 shape 사용
    print('Loaded shape from checkpoint: ', shape.shape)
    
    # OpenCV 윈도우 생성
    window_name = "Real-time Rendering"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, view.image_width, view.image_height)
    
    print("[receiver] 윈도우가 열렸습니다. 렌더링 결과를 표시합니다.")
    print("[receiver] 'q' 키를 누르면 종료됩니다.")
    
    frame_count = 0
    last_fps_time = time.time()
    fps = 0
    
    while True:
        # OpenCV 이벤트 처리 (키 입력 확인)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\n[receiver] 종료합니다.")
            break
        
        events = dict(poller.poll(100))  # 타임아웃을 100ms로 줄여서 반응성 향상
        if sock not in events:
            continue

        # 여기서 "최신만" 남기기 위해 큐를 전부 비움 (drain)
        last_topic, last_buf = None, None
        while True:
            try:
                last_topic, last_buf = sock.recv_multipart(flags=zmq.NOBLOCK)
            except zmq.Again:
                break

        if last_buf is None:
            continue

        try:
            payload = unpack_mano(last_buf)
            print(
                "\n[ok]",
                "frame_id=", payload.get("frame_id"),
                "conf=", payload.get("conf"),
                "go.shape=", payload["global_orient"].shape,
                "hp.shape=", payload["hand_pose"].shape,
                flush=True
            )
            mano_param = load_single_mano_param(payload, shape=shape)
            gaussians.update_mesh_by_param_dict(mano_param)
            rendering = render(view, gaussians, pipeline, background)["render"]
            
            # 렌더링 결과를 numpy 배열로 변환 (H, W, C) 형식, RGB -> BGR 변환
            data = rendering.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
            data_bgr = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
            
            # FPS 계산
            frame_count += 1
            current_time = time.time()
            if current_time - last_fps_time >= 1.0:
                fps = frame_count / (current_time - last_fps_time)
                frame_count = 0
                last_fps_time = current_time
            
            # FPS 텍스트 추가
            if fps > 0:
                cv2.putText(data_bgr, f"FPS: {fps:.1f}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # 윈도우에 이미지 표시
            cv2.imshow(window_name, data_bgr)
            
        except Exception as e:
            print("\n[unpack error]", repr(e), flush=True)
    
    cv2.destroyAllWindows()

        

if __name__ == "__main__":
    ### Usage
    # python render_realtime.py -m output/custom_2_no_loss_masking/
    # Set up command line argument parser
    parser = ArgumentParser(description="Render from checkpoint with MANO parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)

    print("Rendering " + args.model_path)
    # Initialize system state (RNG)
    safe_state(args.quiet)

    main(args, model, pipeline)