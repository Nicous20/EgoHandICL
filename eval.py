import argparse
from pathlib import Path
from re import I
import torch
import json
from mano_param_utils import decode_mano_params
from hamer.models.mano_wrapper import MANO
from hamer.utils.geometry import perspective_projection

def mirror_left_to_right_mano(left_mano_dict):

    """

    """
    def mirror_rotmat_np(rotmat):
        # rotmat: (..., 3, 3)
        M = np.diag([-1.0, 1.0, 1.0])  
        return M @ rotmat @ M.T

    right_mano_dict = left_mano_dict.copy()
    # hand_pose: (15, 3, 3)
    hand_pose = np.array(left_mano_dict['hand_pose'])
    hand_pose_mirrored = mirror_rotmat_np(hand_pose)
    right_mano_dict['hand_pose'] = hand_pose_mirrored.tolist()

    # global_orient: (1, 3, 3)
    global_orient = np.array(left_mano_dict['global_orient'])
    global_orient_mirrored = mirror_rotmat_np(global_orient)
    right_mano_dict['global_orient'] = global_orient_mirrored.tolist()


    right_mano_dict['betas'] = left_mano_dict['betas']
    right_mano_dict['cam'] = left_mano_dict['cam']

    return right_mano_dict


def compute_mano_output(icl_dict, device):
    mano = MANO(


    ).to(device)

    
    pred_mano_params = {}
    pred_mano_params['global_orient'] = torch.tensor([icl_dict['persons']['person_0']['global_orient'].tolist(), icl_dict['persons']['person_1']['global_orient'].tolist()], dtype=torch.float32, device=device)
    pred_mano_params['hand_pose'] = torch.tensor([icl_dict['persons']['person_0']['hand_pose'].tolist(), icl_dict['persons']['person_1']['hand_pose'].tolist()], dtype=torch.float32, device=device)
    pred_mano_params['betas'] = torch.tensor([icl_dict['persons']['person_0']['betas'].tolist(), icl_dict['persons']['person_1']['betas'].tolist()], dtype=torch.float32, device=device)

    pred_cam = torch.tensor([icl_dict['persons']['person_0']['cam'].tolist(), icl_dict['persons']['person_1']['cam'].tolist()], dtype=torch.float32, device=device)
    output = {}
    output['pred_cam'] = pred_cam
    output['pred_mano_params'] = {k: v.clone() for k,v in pred_mano_params.items()}
    device = pred_mano_params['hand_pose'].device
    dtype = pred_mano_params['hand_pose'].dtype
    focal_length = 5000 * torch.ones(2, 2, device=device, dtype=dtype)
    pred_cam_t = torch.stack([pred_cam[:, 1],
                                pred_cam[:, 2],
                                2*focal_length[:, 0]/(256 * pred_cam[:, 0] +1e-9)],dim=-1)
    output['pred_cam_t'] = pred_cam_t
    output['focal_length'] = focal_length
    pred_mano_params['global_orient'] = pred_mano_params['global_orient'].reshape(2, -1, 3, 3)
    pred_mano_params['hand_pose'] = pred_mano_params['hand_pose'].reshape(2, -1, 3, 3)
    pred_mano_params['betas'] = pred_mano_params['betas'].reshape(2, -1)
    mano_output = mano(**{k: v.float() for k,v in pred_mano_params.items()}, pose2rot=False)
    pred_keypoints_3d = mano_output.joints
    pred_vertices = mano_output.vertices
    output['pred_keypoints_3d'] = pred_keypoints_3d.reshape(2, -1, 3)
    output['pred_vertices'] = pred_vertices.reshape(2, -1, 3)
    pred_cam_t = pred_cam_t.reshape(-1, 3)
    focal_length = focal_length.reshape(-1, 2)
    pred_keypoints_2d = perspective_projection(pred_keypoints_3d,
                                                translation=pred_cam_t,
                                                focal_length=focal_length / 256)

    output['pred_keypoints_2d'] = pred_keypoints_2d.reshape(2, -1, 2)

    return output


import numpy as np
    

def torch_mpvpe(predicted, target):
    """
    Mean per-vertex position error using PyTorch.
    Args:
        predicted: [B, V, 3] tensor of predicted vertices
        target: [B, V, 3] tensor of target vertices
    Returns:
        Mean error across all vertices
    """
    return torch.norm(predicted - target, dim=-1).mean()

def torch_p_mpvpe(predicted, target):
    """
    Procrustes-aligned MPVPE
    predicted, target: [B, V, 3]
    """
    assert predicted.shape == target.shape


    muX = torch.mean(target, dim=1, keepdim=True)     # [B,1,3]
    muY = torch.mean(predicted, dim=1, keepdim=True)  # [B,1,3]
    X0 = target - muX
    Y0 = predicted - muY


    normX = torch.sqrt(torch.sum(X0 ** 2, dim=(1, 2), keepdim=True)) + 1e-8
    normY = torch.sqrt(torch.sum(Y0 ** 2, dim=(1, 2), keepdim=True)) + 1e-8


    X0n = X0 / normX
    Y0n = Y0 / normY


    H = torch.matmul(X0n.permute(0, 2, 1), Y0n)        # [B,3,3] = X^T Y
    U, s, Vh = torch.linalg.svd(H)                     # torch>=1.8
    V = Vh.transpose(1, 2)
    R = torch.matmul(V, U.transpose(1, 2))             # R = V U^T


    detR = torch.linalg.det(R)
    mask = (detR < 0).unsqueeze(-1).unsqueeze(-1)      # [B,1,1]
    V_fix = V.clone()
    V_fix[:, :, -1] *= torch.where(mask.squeeze(-1).squeeze(-1), -1.0, 1.0).unsqueeze(-1)
    R = torch.matmul(V_fix, U.transpose(1, 2))


    tr = torch.sum(s, dim=1, keepdim=True).unsqueeze(2)  # [B,1,1]
    a = tr * normX / normY                                # [B,1,1]
    t = muX - a * torch.matmul(muY, R)                    # [B,1,3]


    Y_aligned = a * torch.matmul(predicted, R) + t
    return torch.mean(torch.norm(Y_aligned - target, dim=-1))


import torch

import torch

def torch_mpjpe(predicted, target):
    """
    Mean per-joint position error (MPJPE).
    predicted, target: [B, J, 3] or [J, 3]
    """
    if predicted.ndim == 2:  # [J,3]
        predicted = predicted.unsqueeze(0)
        target = target.unsqueeze(0)
    assert predicted.shape == target.shape
    return torch.norm(predicted - target, dim=-1).mean()


def torch_p_mpjpe(predicted, target):
    """
    Procrustes-aligned MPJPE (Protocol #2).
    predicted, target: [B, J, 3] or [J, 3]
    """
    if predicted.ndim == 2:  # [J,3]
        predicted = predicted.unsqueeze(0)
        target = target.unsqueeze(0)
    assert predicted.shape == target.shape


    muX = target.mean(dim=1, keepdim=True)     # [B,1,3]
    muY = predicted.mean(dim=1, keepdim=True)  # [B,1,3]
    X0 = target - muX
    Y0 = predicted - muY


    normX = torch.sqrt((X0**2).sum(dim=(1,2), keepdim=True)) + 1e-8
    normY = torch.sqrt((Y0**2).sum(dim=(1,2), keepdim=True)) + 1e-8
    X0n = X0 / normX
    Y0n = Y0 / normY


    H = torch.matmul(X0n.transpose(1,2), Y0n)   # [B,3,3]
    U, s, Vh = torch.linalg.svd(H)
    V = Vh.transpose(1,2)
    R = torch.matmul(V, U.transpose(1,2))       # [B,3,3]


    detR = torch.linalg.det(R)
    neg_mask = detR < 0
    if neg_mask.any():
        V[neg_mask, :, -1] *= -1
        R = torch.matmul(V, U.transpose(1,2))


    tr = s.sum(dim=1, keepdim=True).unsqueeze(2)   # [B,1,1]
    a = tr * normX / normY
    t = muX - a * torch.matmul(muY, R)


    Y_aligned = a * torch.matmul(predicted, R) + t
    return torch.norm(Y_aligned - target, dim=-1).mean()



def torch_mrrpe(pred_left, pred_right, gt_left, gt_right, root_idx=0):
    pred_left *= 1000
    pred_right *= 1000
    gt_left *= 1000
    gt_right *= 1000    

    if pred_left.ndim == 2:
        pred_left = pred_left.unsqueeze(0)
        pred_right = pred_right.unsqueeze(0)
        gt_left = gt_left.unsqueeze(0)
        gt_right = gt_right.unsqueeze(0)


    pred_root_L = pred_left[:, root_idx, :]   # [B,3]
    pred_root_R = pred_right[:, root_idx, :]  # [B,3]
    gt_root_L = gt_left[:, root_idx, :]
    gt_root_R = gt_right[:, root_idx, :]


    pred_delta = pred_root_L - pred_root_R    # [B,3]
    gt_delta   = gt_root_L - gt_root_R        # [B,3]


    error = torch.norm(pred_delta - gt_delta, dim=-1)  # [B]
    return error.mean()


def mano_to_tensor(mano_dict):
    import numpy as np
    import copy
    def mirror_right_to_left_mano(right_mano_dict):
        def mirror_rotmat_np(rotmat):
            M = np.diag([-1.0, 1.0, 1.0])
            return M @ rotmat @ M.T
        left_mano_dict = copy.deepcopy(right_mano_dict)
        hand_pose = np.array(right_mano_dict['hand_pose'])
        hand_pose_mirrored = mirror_rotmat_np(hand_pose)
        left_mano_dict['hand_pose'] = hand_pose_mirrored.tolist()
        global_orient = np.array(right_mano_dict['global_orient'])
        global_orient_mirrored = mirror_rotmat_np(global_orient)
        left_mano_dict['global_orient'] = global_orient_mirrored.tolist()
        left_mano_dict['betas'] = right_mano_dict['betas']
        left_mano_dict['cam'] = right_mano_dict['cam']
        return left_mano_dict
    left_hand = None
    right_hand = None
    for person_id in ['person_0', 'person_1']:
        person_data = mano_dict['persons'][person_id]
        if person_data['is_right']:
            right_hand = person_data
        else:
            left_hand = person_data
                
    left_params = torch.cat([
        torch.tensor(left_hand['hand_pose']).flatten(),
        torch.tensor(left_hand['global_orient']).flatten(),
        torch.tensor(left_hand['betas']).flatten(),
        torch.tensor(left_hand['cam']).flatten()
    ])
    right_params = torch.cat([
        torch.tensor(right_hand['hand_pose']).flatten(),
        torch.tensor(right_hand['global_orient']).flatten(),
        torch.tensor(right_hand['betas']).flatten(),
        torch.tensor(right_hand['cam']).flatten()
    ])
    return torch.cat([left_params, right_params])

from handicl.loss_utils import compute_hands_l2_loss_from_dicts
from tqdm import tqdm


import cv2
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', type=str, required=True, help='Directory with images')
    parser.add_argument('--mano_wilor_dir', type=str, required=True, help='Directory with mano_wilor json')
    parser.add_argument('--mano_gt_dir', type=str, required=True, help='Directory with mano_gt json')
    parser.add_argument('--ehi', action='store_true', help='Eval EgoHandICL')
    
    # parser.add_argument('--ext', type=str, default='jpg', help='Image file extension')
    args = parser.parse_args()
    eval_ehi = args.ehi
    print(eval_ehi)

    # img_dir = Path(args.img_dir)
    # img_dir is a json file with img paths
    with open(args.img_dir, 'r') as f:
        img_paths = json.load(f)
    

    img_paths = [Path(p) for p in img_paths]
    # import pdb; pdb.set_trace()
    mano_wilor_dir = Path(args.mano_wilor_dir)
    mano_gt_dir = Path(args.mano_gt_dir)
    # img_paths = sorted(list(img_dir.glob(f'*.{args.ext}')))

    left_num = 0
    right_num = 0
    both_num = 0
    total_left_mpjpe = 0
    total_right_mpjpe = 0
    total_p_mpjpe = 0
    total_p_mpvpe = 0
    total_both_p_mpjpe = 0
    total_both_p_mpvpe = 0
    total_mrrpe = 0 
    count = 0


    f5_count = 0
    f15_count = 0

    val_list = []
    num_ego_l = num_ego_r = num_wilor_l = num_wilor_r = 0
    both_mpjpe = 0
    img_dict = {} 
    num = 0
    for img_path in tqdm(img_paths):

        try:
                

            
            mano_hamer_dir = Path('')
            mano_hamer_path = mano_hamer_dir / f"{img_path.stem}_mano.json"
            mano_wilor_path = mano_wilor_dir / f"{img_path.stem}_wilor.json"
            gt_path = mano_gt_dir / f"{img_path.stem}_gt.json"
            if not mano_wilor_path.exists() or not gt_path.exists():
                # print(f"Missing: {mano_wilor_path if not mano_wilor_path.exists() else gt_path}, skip.")
                continue
            with open(mano_wilor_path, 'r') as f:
                pred_mano = json.load(f)
            
            
            with open(gt_path, 'r') as f:
                gt_mano = json.load(f)
            
            egohand_icl_path = img_path.parent / f"{img_path.stem}_infer_v7-2.json"    


            # pred_tensor = mano_to_tensor(pred_mano)
            gt_tensor = mano_to_tensor(gt_mano)
            
            left_hand = None
            right_hand = None
        
            for item in pred_mano['persons']:
                if pred_mano['persons'][item]['is_right'] < 1e-5 and left_hand is None:
                    left_hand = pred_mano['persons'][item]
                    left_kp = torch.tensor(left_hand['pred_keypoints_3d'])
                    left_vt = torch.tensor(left_hand['pred_vertices'])
                    left_kp *= torch.tensor([-1, 1, 1])
                    left_vt *= torch.tensor([-1, 1, 1])
                elif pred_mano['persons'][item]['is_right'] > 1e-5 and right_hand is None:
                    right_hand = pred_mano['persons'][item]
                    right_kp = torch.tensor(right_hand['pred_keypoints_3d'])
                    right_vt = torch.tensor(right_hand['pred_vertices'])
                    
            if eval_ehi and left_hand is None and right_hand is not None:
                with open(mano_hamer_path, 'r') as f:
                    hamer_mano = json.load(f)
                        
            if eval_ehi and left_hand is not None and right_hand is not None and torch_p_mpvpe(left_vt.reshape(1, -1, 3), right_vt.reshape(1, -1, 3)).item() < 0.005:
                # left_hand = None
                right_hand = None
                    
                   

            gt_dict = decode_mano_params(gt_tensor)

            gt_mano_output = compute_mano_output(gt_dict, device=gt_tensor.device)

            # Test EgoHandICL
            
            if eval_ehi and left_hand is not None and right_hand is not None:
                # import pdb; pdb.set_trace()
                if egohand_icl_path.exists() is not True:
                    continue
                # import pdb; pdb.set_trace()
                with open(egohand_icl_path, 'r') as f:
                    egohand_icl_mano = json.load(f)
                
                egohand_icl_tensor = mano_to_tensor(egohand_icl_mano)
                egohand_icl_dict = decode_mano_params(egohand_icl_tensor)
                egohand_icl_output = compute_mano_output(egohand_icl_dict, device=egohand_icl_tensor.device)
                left_kp = egohand_icl_output['pred_keypoints_3d'][0]
                right_kp = egohand_icl_output['pred_keypoints_3d'][1]
                left_vt = egohand_icl_output['pred_vertices'][0]
                right_vt = egohand_icl_output['pred_vertices'][1]

                # rigth_kp = torch.tensor(right_hand['pred_keypoints_3d'])
                # right_vt = torch.tensor(right_hand['pred_vertices'])

                mpvpe_ehi_l = torch_p_mpvpe(left_vt.reshape(1, -1, 3), gt_mano_output['pred_vertices'][0].reshape(1, -1, 3))
                mpvpe_ehi_r = torch_p_mpvpe(right_vt.reshape(1, -1, 3), gt_mano_output['pred_vertices'][1].reshape(1, -1, 3))
                mpvpe_wilor_l = torch_p_mpvpe(torch.tensor(left_hand['pred_vertices'])* torch.tensor([-1, 1, 1]).reshape(1, -1, 3), gt_mano_output['pred_vertices'][0].reshape(1, -1, 3))
                mpvpe_wilor_r = torch_p_mpvpe(torch.tensor(right_hand['pred_vertices']).reshape(1, -1, 3), gt_mano_output['pred_vertices'][1].reshape(1, -1, 3))
                # import pdb; pdb.set_trace()
                if mpvpe_ehi_l < mpvpe_wilor_l and mpvpe_ehi_r < mpvpe_wilor_r:
                    
                    imp = (mpvpe_wilor_l + mpvpe_wilor_r) - (mpvpe_ehi_l + mpvpe_ehi_r)
                    ratio = imp / (mpvpe_wilor_l + mpvpe_wilor_r)
                    if ratio > 0.7:
                        img_dict[str(img_path)] = ratio.item()
                        num += 1 
                        import pdb; pdb.set_trace()
                if mpvpe_ehi_l < mpvpe_wilor_l:
                    num_ego_l +=1
                else:
                    num_wilor_l += 1
                    # left_kp = torch.tensor(left_hand['pred_keypoints_3d']) * torch.tensor([-1, 1, 1])
                    # left_vt = torch.tensor(left_hand['pred_vertices']) * torch.tensor([-1, 1, 1])
                
                if mpvpe_ehi_r < mpvpe_wilor_r:
                    num_ego_r +=1
                else:
                    num_wilor_r += 1
                    # rigth_kp = torch.tensor(right_hand['pred_keypoints_3d'])
                    # right_vt = torch.tensor(right_hand['pred_vertices'])
                

                total_p_mpjpe += torch_p_mpjpe(left_kp.reshape(1, -1, 3), gt_mano_output['pred_keypoints_3d'][0].reshape(1, -1, 3))
                total_p_mpjpe += torch_p_mpjpe(right_kp.reshape(1, -1, 3), gt_mano_output['pred_keypoints_3d'][1].reshape(1, -1, 3))
                pmpvpe = torch_p_mpvpe(left_vt.reshape(1, -1, 3), gt_mano_output['pred_vertices'][0].reshape(1, -1, 3))
                total_p_mpvpe += pmpvpe
                if pmpvpe.item() < 0.005:
                    f5_count += 1
                if pmpvpe.item() < 0.015:
                    f15_count += 1
                # else:
                #     print('l', img_path, torch_p_mpvpe(left_vt.reshape(1, -1, 3), right_vt.reshape(1, -1, 3)).item())
                pmpvpe = torch_p_mpvpe(right_vt.reshape(1, -1, 3), gt_mano_output['pred_vertices'][1].reshape(1, -1, 3))
                total_p_mpvpe += pmpvpe
                if pmpvpe.item() < 0.005:
                    f5_count += 1
                if pmpvpe.item() < 0.015:
                    f15_count += 1
                # else:
                #     print('r', img_path, torch_p_mpvpe(left_vt.reshape(1, -1, 3), right_vt.reshape(1, -1, 3)).item())
                
                total_both_p_mpjpe += (torch_p_mpjpe(left_kp.reshape(1, -1, 3), gt_mano_output['pred_keypoints_3d'][0].reshape(1, -1, 3)) + torch_p_mpjpe(right_kp.reshape(1, -1, 3), gt_mano_output['pred_keypoints_3d'][1].reshape(1, -1, 3))) / 2
                total_both_p_mpvpe += (torch_p_mpvpe(left_vt.reshape(1, -1, 3), gt_mano_output['pred_vertices'][0].reshape(1, -1, 3)) + torch_p_mpvpe(right_vt.reshape(1, -1, 3), gt_mano_output['pred_vertices'][1].reshape(1, -1, 3))) / 2
                total_mrrpe += torch_mrrpe(left_kp.reshape(1, -1, 3), right_kp.reshape(1, -1, 3), gt_mano_output['pred_keypoints_3d'][0].reshape(1, -1, 3), gt_mano_output['pred_keypoints_3d'][1].reshape(1, -1, 3))
                left_num += 1
                right_num += 1
                both_num += 1
                

               

                continue
            
            if left_hand is not None:
                left_num += 1
                total_p_mpjpe += torch_p_mpjpe(left_kp.reshape(1, -1, 3), gt_mano_output['pred_keypoints_3d'][0].reshape(1, -1, 3))
                pmpvpe = torch_p_mpvpe(left_vt.reshape(1, -1, 3), gt_mano_output['pred_vertices'][0].reshape(1, -1, 3))
                # print(pmpvpe.item())
                total_p_mpvpe += pmpvpe
                if pmpvpe.item() < 0.005:
                    f5_count += 1
                if pmpvpe.item() < 0.015:
                    f15_count += 1
                # else:
                #     import pdb; pdb.set_trace()
                #     print('wl', img_path, torch_p_mpvpe(left_vt.reshape(1, -1, 3), right_vt.reshape(1, -1, 3)).item())
            if right_hand is not None:
                right_num += 1
                total_p_mpjpe += torch_p_mpjpe(right_kp.reshape(1, -1, 3), gt_mano_output['pred_keypoints_3d'][1].reshape(1, -1, 3))
                pmpvpe = torch_p_mpvpe(right_vt.reshape(1, -1, 3), gt_mano_output['pred_vertices'][1].reshape(1, -1, 3))
                # print(pmpvpe.item())
                total_p_mpvpe += pmpvpe
                if pmpvpe.item() < 0.005:
                    f5_count += 1
                if pmpvpe.item() < 0.015:
                    f15_count += 1
                # else:
                #     print('wr', img_path, torch_p_mpvpe(left_vt.reshape(1, -1, 3), right_vt.reshape(1, -1, 3)).item())
            
            # import pdb; pdb.set_trace()
            if left_hand is not None and right_hand is not None:
                total_mrrpe += torch_mrrpe(left_kp.reshape(1, -1, 3), right_kp.reshape(1, -1, 3), gt_mano_output['pred_keypoints_3d'][0].reshape(1, -1, 3), gt_mano_output['pred_keypoints_3d'][1].reshape(1, -1, 3))
                both_num += 1
                total_both_p_mpjpe += (torch_p_mpjpe(left_kp.reshape(1, -1, 3), gt_mano_output['pred_keypoints_3d'][0].reshape(1, -1, 3)) + torch_p_mpjpe(right_kp.reshape(1, -1, 3), gt_mano_output['pred_keypoints_3d'][1].reshape(1, -1, 3))) / 2
                total_both_p_mpvpe += (torch_p_mpvpe(left_vt.reshape(1, -1, 3), gt_mano_output['pred_vertices'][0].reshape(1, -1, 3)) + torch_p_mpvpe(right_vt.reshape(1, -1, 3), gt_mano_output['pred_vertices'][1].reshape(1, -1, 3))) / 2
            

        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
            continue
    print(f"Avg P-MPJPE: {total_p_mpjpe / (left_num + right_num):.6f} over {left_num + right_num} samples.")
    print(f"Avg P-MPVPE: {total_p_mpvpe / (left_num + right_num):.6f} over {left_num + right_num} samples.")
    print(f"Avg F5: {f5_count / (left_num + right_num):.6f} over {left_num + right_num}  samples.")
    print(f"Avg F15: {f15_count / (left_num + right_num):.6f} over {left_num + right_num}  samples.")
    print(f"For Both Hand:")
    print(f"Avg P-MPJPE: {total_both_p_mpjpe / both_num:.6f} over {both_num} samples.")
    print(f"Avg P-MPVPE: {total_both_p_mpvpe / both_num:.6f} over {both_num} samples.")
    print(f"Avg MRRPE: {total_mrrpe / both_num:.6f} over {both_num} samples.")
    
    print(num_ego_l, num_ego_r, num_wilor_l, num_wilor_r)
    imp_json_save_path = ''
    with open(imp_json_save_path, 'w') as f:
        json.dump(img_dict, f)

if __name__ == '__main__':
    main()  