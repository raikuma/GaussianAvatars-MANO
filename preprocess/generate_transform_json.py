import sys, json, math, glob, os
from PIL import Image

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

root_path = sys.argv[1]
param_path = f"{root_path}/mano_params/debug_persp_frames"

frames = []
for idx in os.listdir(param_path):
    json_path = os.path.join(param_path, idx, 'stats.json')
    if os.path.exists(json_path):
        with open(json_path) as json_file:
            stats = json.load(json_file)
        
        frame = stats.get("frame")
        i = int(frame)

        if i < 0:
            continue

        h, w = stats.get("image_hw")
        cx, cy = stats.get("ortho_princpt")
        fl_x, fl_y = stats.get("persp_fx_fy")
        fovx = focal2fov(fl_x, w)
        fovy = focal2fov(fl_y, h)

        frame_data = {
            "timestep_index": i,
            "timestep_index_original": i,
            "timestep_id": f"frame_{i*3:05d}",
            "camera_index": i,
            "camera_id": f"cam_{i:05d}",
            "cx": cx,
            "cy": cy,
            "h": h,
            "w": w,
            "camera_angle_x": fovx,
            "camera_angle_y": fovy,
            "transform_matrix": [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0]
            ],
            "file_path": f"frames/{frame}.png",
            "fg_mask_path": f"masks/{frame}.png",
            "mano_mesh_path": f"mano_params/meshes/{frame}.obj",
            "mano_param_path": f"mano_params/params/{frame}.json",
        }

        frames.append(frame_data)

frames.sort(key=lambda x:x['timestep_index'])

json_data = {
    "frames": frames
}

for split in ["train", "val", "test"]:
    with open(f"{root_path}/transforms_{split}.json", "w") as f:
        json.dump(json_data, f, indent=4)