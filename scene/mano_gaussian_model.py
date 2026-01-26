# 
# Toyota Motor Europe NV/SA and its affiliated companies retain all intellectual 
# property and proprietary rights in and to this software and related documentation. 
# Any commercial use, reproduction, disclosure or distribution of this software and 
# related documentation without an express license agreement from Toyota Motor Europe NV/SA 
# is strictly prohibited.
#

from pathlib import Path
import numpy as np
import torch
import smplx

from .gaussian_model import GaussianModel
from utils.graphics_utils import compute_face_orientation
# from pytorch3d.transforms import matrix_to_quaternion
from roma import rotmat_to_unitquat, quat_xyzw_to_wxyz

MANO_MODEL_PATH = 'mano_model/MANO_RIGHT.pkl'

class ManoGaussianModel(GaussianModel):
    def __init__(self, sh_degree : int, not_finetune_mano_params=False):
        super().__init__(sh_degree)

        self.not_finetune_mano_params = not_finetune_mano_params

        layer_arg = {'create_global_orient': False, 'create_hand_pose': False, 'create_betas': False, 'create_transl': False}
        self.mano_model = smplx.create(
            MANO_MODEL_PATH, 'mano', is_rhand=True, use_pca=False, flat_hand_mean=False, **layer_arg
        ).cuda()
        self.mano_param = None
        self.mano_param_orig = None

        # binding is initialized once the mesh topology is known
        if self.binding is None:
            self.binding = torch.arange(len(self.mano_model.faces)).cuda()
            self.binding_counter = torch.ones(len(self.mano_model.faces), dtype=torch.int32).cuda()

    def load_meshes(self, train_meshes, test_meshes, tgt_train_meshes, tgt_test_meshes, 
                    fix_root_rotation=False, fix_root_translation=False, fix_hand_pose=False):
        # if self.flame_param is None:
        if self.mano_param is None:
            meshes = {**train_meshes, **test_meshes}
            tgt_meshes = {**tgt_train_meshes, **tgt_test_meshes}
            pose_meshes = meshes if len(tgt_meshes) == 0 else tgt_meshes
            
            self.num_timesteps = max(pose_meshes) + 1  # required by viewers
            num_verts = self.mano_model.v_template.shape[0]

            T = self.num_timesteps

            self.mano_param = {
                'shape': torch.from_numpy(np.array(meshes[0]['shape'])),
                'root_pose': torch.zeros([T, 3]),
                'root_trans': torch.zeros([T, 3]),
                'hand_pose': torch.zeros([T, len(meshes[0]['hand_pose'])]),
                'static_offset': torch.zeros([num_verts, 3]),
                'dynamic_offset': torch.zeros([T, num_verts, 3]),
            }

            # Get source's first timestep values for root_pose, root_trans, and hand_pose (if fix options are enabled)
            source_first_timestep = min(meshes.keys()) if len(meshes) > 0 else 0
            source_root_pose = None
            source_root_trans = None
            source_hand_pose = None
            if fix_root_rotation or fix_root_translation:
                source_root_pose = torch.from_numpy(np.array(meshes[source_first_timestep]['root_pose']))
                source_root_trans = torch.from_numpy(np.array(meshes[source_first_timestep]['root_trans']))
            if fix_hand_pose:
                source_hand_pose = torch.from_numpy(np.array(meshes[source_first_timestep]['hand_pose']))
            
            # Load parameters from pose_meshes (target if available, otherwise source)
            for i, mesh in pose_meshes.items():
                # Use fixed source values if options are enabled, otherwise use from pose_meshes
                if fix_root_rotation:
                    self.mano_param['root_pose'][i] = source_root_pose.clone()
                else:
                    self.mano_param['root_pose'][i] = torch.from_numpy(np.array(mesh['root_pose']))
                
                if fix_root_translation:
                    self.mano_param['root_trans'][i] = source_root_trans.clone()
                else:
                    self.mano_param['root_trans'][i] = torch.from_numpy(np.array(mesh['root_trans']))
                
                if fix_hand_pose:
                    self.mano_param['hand_pose'][i] = source_hand_pose.clone()
                else:
                    self.mano_param['hand_pose'][i] = torch.from_numpy(np.array(mesh['hand_pose']))
                # self.mano_param['dynamic_offset'][i] = torch.from_numpy(np.array(mesh['dynamic_offset']))

            for k, v in self.mano_param.items():
                self.mano_param[k] = v.float().cuda()
            
            # self.flame_param_orig = {k: v.clone() for k, v in self.flame_param.items()}

            self.mano_param_orig = {k: v.clone() for k, v in self.mano_param.items()}
        else:
            # NOTE: not sure when this happens
            import ipdb; ipdb.set_trace()
            pass
    
    def update_mesh_by_param_dict(self, mano_param):
        if 'shape' in mano_param:
            shape = mano_param['shape']
        else:
            shape = self.mano_param['shape']

        if 'static_offset' in mano_param:
            static_offset = mano_param['static_offset']
        else:
            static_offset = self.mano_param['static_offset']

        verts = self.mano_model(
            betas=shape[None, ...],
            global_orient=mano_param['root_pose'].cuda(),
            transl=mano_param['root_trans'].cuda(),
            hand_pose=mano_param['hand_pose'].cuda(),
        ).vertices
        verts_cano = self.mano_model(
            betas=mano_param['shape'][None, ...],
            global_orient=torch.zeros([1, 3]).cuda(),
            transl=torch.zeros([1, 3]).cuda(),
            hand_pose=torch.zeros([1, len(mano_param['hand_pose'][0])]).cuda(),
        ).vertices
        self.update_mesh_properties(verts, verts_cano)

    def select_mesh_by_timestep(self, timestep, original=False, zero_origin=False):
        self.timestep = timestep
        mano_param = self.mano_param_orig if original and self.mano_param_orig is not None else self.mano_param

        if zero_origin:
            verts = self.mano_model(
                betas=mano_param['shape'][None, ...],
                global_orient=mano_param['root_pose'][[timestep]],
                hand_pose=mano_param['hand_pose'][[timestep]],
            ).vertices
            verts_cano = self.mano_model(
                betas=mano_param['shape'][None, ...],
                global_orient=torch.zeros([1, 3]).cuda(),
                hand_pose=torch.zeros([1, len(mano_param['hand_pose'][0])]).cuda(),
            ).vertices
        else:
            verts = self.mano_model(
                betas=mano_param['shape'][None, ...],
                global_orient=mano_param['root_pose'][[timestep]],
                transl=mano_param['root_trans'][[timestep]],
                hand_pose=mano_param['hand_pose'][[timestep]],
            ).vertices
            verts_cano = self.mano_model(
                betas=mano_param['shape'][None, ...],
                global_orient=torch.zeros([1, 3]).cuda(),
                transl=torch.zeros([1, 3]).cuda(),
                hand_pose=torch.zeros([1, len(mano_param['hand_pose'][0])]).cuda(),
            ).vertices
        self.update_mesh_properties(verts, verts_cano)
    
    def update_mesh_properties(self, verts, verts_cano):
        # faces = self.mano_model.faces
        faces = torch.tensor(self.mano_model.faces.astype(np.int64), dtype=torch.int64, device=verts.device)
        triangles = verts[:, faces]

        # position
        self.face_center = triangles.mean(dim=-2).squeeze(0)

        # orientation and scale
        self.face_orien_mat, self.face_scaling = compute_face_orientation(verts.squeeze(0), faces.squeeze(0), return_scale=True)
        # self.face_orien_quat = matrix_to_quaternion(self.face_orien_mat)  # pytorch3d (WXYZ)
        self.face_orien_quat = quat_xyzw_to_wxyz(rotmat_to_unitquat(self.face_orien_mat))  # roma

        # for mesh rendering
        self.verts = verts
        self.faces = faces

        # for mesh regularization
        self.verts_cano = verts_cano
    
    def compute_dynamic_offset_loss(self):
        # loss_dynamic = (self.flame_param['dynamic_offset'][[self.timestep]] - self.mano_model_orig['dynamic_offset'][[self.timestep]]).norm(dim=-1)
        loss_dynamic = self.flame_param['dynamic_offset'][[self.timestep]].norm(dim=-1)
        return loss_dynamic.mean()
    
    def compute_laplacian_loss(self):
        # offset = self.mano_param['static_offset'] + self.mano_param['dynamic_offset'][[self.timestep]]
        offset = self.mano_param['dynamic_offset'][[self.timestep]]
        verts_wo_offset = (self.verts_cano - offset).detach()
        verts_w_offset = verts_wo_offset + offset

        L = self.mano_model.laplacian_matrix[None, ...].detach()  # (1, V, V) # TODO

        lap_wo = L.bmm(verts_wo_offset).detach()
        lap_w = L.bmm(verts_w_offset)
        diff = (lap_wo - lap_w) ** 2
        diff = diff.sum(dim=-1, keepdim=True)
        return diff.mean()
    
    def training_setup(self, training_args):
        super().training_setup(training_args)

        if self.not_finetune_mano_params:
            return

        # # shape
        # self.mano_param['shape'].requires_grad = True
        # param_shape = {'params': [self.mano_param['shape']], 'lr': 1e-5, "name": "shape"}
        # self.optimizer.add_param_group(param_shape)

        # pose
        self.mano_param['hand_pose'].requires_grad = True
        params = [
            self.mano_param['hand_pose'],
        ]
        param_pose = {'params': params, 'lr': training_args.mano_pose_lr, "name": "pose"}
        self.optimizer.add_param_group(param_pose)

        # translation
        self.mano_param['root_trans'].requires_grad = True
        param_trans = {'params': [self.mano_param['root_trans']], 'lr': training_args.mano_trans_lr, "name": "trans"}
        self.optimizer.add_param_group(param_trans)

        # # static_offset
        # self.mano_param['static_offset'].requires_grad = True
        # param_static_offset = {'params': [self.mano_param['static_offset']], 'lr': 1e-6, "name": "static_offset"}
        # self.optimizer.add_param_group(param_static_offset)

        # # dynamic_offset
        # self.mano_param['dynamic_offset'].requires_grad = True
        # param_dynamic_offset = {'params': [self.mano_param['dynamic_offset']], 'lr': 1.6e-6, "name": "dynamic_offset"}
        # self.optimizer.add_param_group(param_dynamic_offset)

    def save_ply(self, path):
        super().save_ply(path)

        npz_path = Path(path).parent / "mano_param.npz"
        mano_param = {k: v.cpu().numpy() for k, v in self.mano_param.items()}
        np.savez(str(npz_path), **mano_param)

    def load_ply(self, path, **kwargs):
        super().load_ply(path)

        if not kwargs.get('has_target', False):
            npz_path = Path(path).parent / "mano_param.npz"
            mano_param = np.load(str(npz_path))
            mano_param = {k: torch.from_numpy(v).cuda() for k, v in mano_param.items()}
            
            self.mano_param = mano_param
            self.num_timesteps = self.mano_param['hand_pose'].shape[0]

        if 'motion_path' in kwargs and kwargs['motion_path'] is not None:
            motion_path = Path(kwargs['motion_path'])
            mano_param = np.load(str(motion_path))
            mano_param = {k: torch.from_numpy(v).cuda() for k, v in mano_param.items() if v.dtype == np.float32}
            
            self.mano_param = {
                'shape': self.mano_param['shape'],
                'static_offset': self.mano_param['static_offset'],
                'root_pose': mano_param['root_pose'],
                'root_trans': mano_param['root_trans'],
                'hand_pose': mano_param['hand_pose'],
                'dynamic_offset': mano_param['dynamic_offset'],
            }
            self.num_timesteps = self.mano_param['hand_pose'].shape[0]  # required by viewers
        
        if 'disable_fid' in kwargs and len(kwargs['disable_fid']) > 0:
            mask = (self.binding[:, None] != kwargs['disable_fid'][None, :]).all(-1)

            self.binding = self.binding[mask]
            self._xyz = self._xyz[mask]
            self._features_dc = self._features_dc[mask]
            self._features_rest = self._features_rest[mask]
            self._scaling = self._scaling[mask]
            self._rotation = self._rotation[mask]
            self._opacity = self._opacity[mask]
