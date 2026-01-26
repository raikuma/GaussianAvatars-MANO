# mano_stream_protocol.py
import time
import numpy as np
import msgpack
import msgpack_numpy as m

m.patch()

def pack_mano(
    global_orient, hand_pose, betas=None, transl=None,
    cam=None, conf=1.0, handedness="right", frame_id=0
):
    """
    All arrays should be numpy float32.
    global_orient: (3,) axis-angle  (or (3,3) rotmat - 아래에서 설명)
    hand_pose:     (45,) axis-angle (15 joints * 3)
    betas:         (10,) optional
    transl:        (3,) optional
    cam: dict optional (e.g., {"fx":..,"fy":..,"cx":..,"cy":..,"W":..,"H":..} or ortho params)
    """
    def _to_np(x):
        if x is None: return None
        if hasattr(x, "detach"): x = x.detach().cpu().numpy()
        x = np.asarray(x)
        if x.dtype != np.float32: x = x.astype(np.float32)
        return x

    payload = {
        "t": time.time(),
        "frame_id": int(frame_id),
        "handedness": handedness,
        "conf": float(conf),
        "global_orient": _to_np(global_orient),
        "hand_pose": _to_np(hand_pose),
        "betas": _to_np(betas),
        "transl": _to_np(transl),
        "cam": cam,
    }
    return msgpack.packb(payload, use_bin_type=True)

def unpack_mano(buf):
    payload = msgpack.unpackb(buf, raw=False)
    return payload