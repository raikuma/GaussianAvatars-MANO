import sys, json
root_path = sys.argv[1]

cx = 320.0
cy = 180.0
# fl_x = 2000.0
# fl_y = 2000.0
w = 640
h = 360
camera_angle_x = 0.3173
# camera_angle_y = 0.1795
# camera_angle_x = 0.0610865
# camera_angle_y = 0.1795

N = 53

json_data = {
    "cx": cx,
    "cy": cy,
    # "fl_x": fl_x,
    # "fl_y": fl_y,
    "w": w,
    "h": h,
    "camera_angle_x": camera_angle_x,
    # "camera_angle_y": camera_angle_y,
    "frames": []
}

for i in range(N):
    frame_data = {
        "timestep_index": i,
        "timestep_index_original": i,
        "timestep_id": f"frame_{i*3:05d}",
        "camera_index": i,
        "camera_id": f"cam_{i:05d}",
        "cx": cx,
        "cy": cy,
        # "fl_x": fl_x,
        # "fl_y": fl_y,
        "h": h,
        "w": w,
        "camera_angle_x": camera_angle_x,
        # "camera_angle_y": camera_angle_y,
        "transform_matrix": [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ],
        "file_path": f"frames/{i}.png",
        "fg_mask_path": f"masks/{i}.png",
        "mano_mesh_path": f"mano_params/meshes/{i}.obj",
        "mano_param_path": f"mano_params/params/{i}.json"
    }
    json_data["frames"].append(frame_data)

for split in ["train", "val", "test"]:
    with open(f"{root_path}/transforms_{split}.json", "w") as f:
        json.dump(json_data, f, indent=4)