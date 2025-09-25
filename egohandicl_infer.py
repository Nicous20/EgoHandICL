#!/usr/bin/env python3
import torch
from pathlib import Path
from torchvision import transforms
from PIL import Image
import argparse
import json
import random
import numpy as np
import cv2


from handicl.egohandicl import HandICL  



def mirror_right_to_left_mano(right_mano_dict: dict) -> dict:

    def mirror_rotmat_np(rotmat: np.ndarray) -> np.ndarray:
        M = np.diag([-1.0, 1.0, 1.0]) 
        return M @ rotmat @ M.T

    left_mano_dict = json.loads(json.dumps(right_mano_dict))  
    hand_pose = np.array(right_mano_dict['hand_pose'])        # (15,3,3)
    global_orient = np.array(right_mano_dict['global_orient'])# (1,3,3)
    left_mano_dict['hand_pose'] = mirror_rotmat_np(hand_pose).tolist()
    left_mano_dict['global_orient'] = mirror_rotmat_np(global_orient).tolist()

    left_mano_dict['betas'] = right_mano_dict['betas']
    left_mano_dict['cam'] = right_mano_dict['cam']
    return left_mano_dict

def fill_hand(mano_dict, img_path, data_root):

    mano_hamer_dir = Path(data_root) / 'mano_hamer'
    mano_hamer_path = mano_hamer_dir / f"{Path(img_path).stem}_mano.json"
    with open(mano_hamer_path, 'r') as f:
        mano_hamer = json.load(f)
    mano_left_hand = None
    mano_right_hand = None
    for pid in mano_hamer['persons']:
        if mano_hamer['persons'][pid]['is_right']:
            mano_right_hand = mano_hamer['persons'][pid]
        else:
            mano_left_hand = mano_hamer['persons'][pid]

    left_hand, right_hand = None, None
    for pid in mano_dict['persons']:

        is_right_val = mano_dict['persons'][pid]['is_right']
        if isinstance(is_right_val, (int, float)):
            is_left = (is_right_val < 1e-5)
        else:
            is_left = not bool(is_right_val)
        if is_left:
            left_hand = mano_dict['persons'][pid]
        else:
            right_hand = mano_dict['persons'][pid]

    if left_hand is None:  left_hand  = mano_left_hand
    if right_hand is None: right_hand = mano_right_hand

    return {
        'persons': {
            'person_0': left_hand,
            'person_1': right_hand
        }
    }

def rodrigues_R_to_aa_list(R_list):
    # R_list: list of (3,3)
    aa = []
    for R in R_list:
        rvec, _ = cv2.Rodrigues(np.array(R))
        aa.append(rvec.flatten().tolist())  # 3
    return aa  # list of [3]

def process_mano(mano_dict):
    left = mano_dict['persons']['person_0']
    right = mano_dict['persons']['person_1']
    import pdb; pdb.set_trace()

    left_go_aa  = rodrigues_R_to_aa_list(left['global_orient'])    # [[3]]
    left_hp_aa  = rodrigues_R_to_aa_list(left['hand_pose'])        # 15*[3]
    right_go_aa = rodrigues_R_to_aa_list(right['global_orient'])   # [[3]]
    right_hp_aa = rodrigues_R_to_aa_list(right['hand_pose'])       # 15*[3]

    left_paras_list = left_go_aa + left_hp_aa + left['betas'] + left['cam']
    right_paras_list = right_go_aa + right_hp_aa + right['betas'] + right['cam']
    

    return torch.cat([left_paras_list, right_paras_list], dim=0).to(torch.float32)
    pose_list = (
        left_go_aa + left_hp_aa + right_go_aa + right_hp_aa  
    )
    pose_params = torch.tensor(np.array(pose_list).reshape(-1))     # 96

    betas_params = torch.tensor(left['betas'] + right['betas']).flatten()  # 20
    cam_params   = torch.tensor(left['cam'] + right['cam']).flatten()      # 6

    return torch.cat([pose_params, betas_params, cam_params], dim=0).to(torch.float32)       # 122

def get_mano_pair_from_paths(img_path, data_root, is_template):

    img_path = Path(img_path)
    if is_template:
        pred_path = Path(data_root) / 'mano_wilor' / f"{img_path.stem}_wilor.json"
        gt_path   = Path(data_root) / 'mano_gt'    / f"{img_path.stem}_gt.json"

        with open(pred_path, 'r') as f:
            pred_mano = json.load(f)
        with open(gt_path, 'r') as f:
            gt_mano = json.load(f)
        return pred_mano, gt_mano
    else:
        pred_path = Path(data_root) / 'mano_wilor' / f"{img_path.stem}_wilor.json"
        with open(pred_path, 'r') as f:
            pred_mano = json.load(f)
        return pred_mano, None



def build_model(ckp_path, hidden_dim=768, num_heads=12):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HandICL({'hidden_dim': hidden_dim, 'num_heads': num_heads}).to(device)
    # ckpt = torch.load(ckp_path, map_location=device)
    ckpt = torch.load(ckp_path, map_location="cpu") 
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model, device

def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

@torch.no_grad()
def infer_one(img_path, ckp_path, data_root, json_path,
              hidden_dim=768, num_heads=12,
              mask_ratio=0.7,
              text_instruction_path=None,
              use_amp=True):

    model, device = build_model(ckp_path, hidden_dim, num_heads)
    tfm = get_transform()


    with open(json_path, 'r') as f:
        candidates = json.load(f)
    template_img_path = Path(random.choice(candidates))


    if text_instruction_path is None:

        text_instruction_path = str(Path(data_root) / 'text_instruction.json')
    with open(text_instruction_path, 'r') as f:
        text_instruction = json.load(f)


    img1 = tfm(Image.open(template_img_path).convert('RGB')).unsqueeze(0).to(device)
    img2 = tfm(Image.open(img_path).convert('RGB')).unsqueeze(0).to(device)


    m1a_raw, m1b_raw = get_mano_pair_from_paths(template_img_path, data_root, is_template=True)
    m2a_raw, _       = get_mano_pair_from_paths(img_path, data_root, is_template=False) 


    m1a_filled = fill_hand(m1a_raw, template_img_path, data_root)
    m1b_filled = fill_hand(m1b_raw, template_img_path, data_root)
    m2a_filled = fill_hand(m2a_raw, img_path,           data_root)


    import pdb; pdb.set_trace()
    m1a = process_mano(m1a_filled)  # [122]
    m1b = process_mano(m1b_filled)  # [122]
    m2a = process_mano(m2a_filled)  # [122]


    m2b = torch.randn_like(m2a) * 1e-5

    batch = {
        'type': Path(json_path).name.split('_')[-2] if '_' in Path(json_path).name else '0',
        'img1': img1,
        'img2': img2,
        'text1': text_instruction[template_img_path.stem],
        'text2': text_instruction[Path(img_path).stem],
        'mano_params': {
            'm1a': m1a.unsqueeze(0).to(device),
            'm1b': m1b.unsqueeze(0).to(device),
            'm2a': m2a.unsqueeze(0).to(device),
            'm2b': m2b.unsqueeze(0).to(device)
        },
        'mask_ratio': mask_ratio

    }

    if use_amp:
        with torch.cuda.amp.autocast():
            output = model(batch)  
    else:
        output = model(batch)

    out = output[0] if output.dim() == 2 else output   
    single_dim = 122
    m2b_pred_vec = out[-single_dim:].detach().cpu()     # [122]


    def aa_to_R_list(flat_aa, n):
        arr = flat_aa.reshape(n, 3).numpy()
        R_list = []
        for i in range(n):
            R, _ = cv2.Rodrigues(arr[i])
            R_list.append(R.tolist())
        return R_list

    i = 0
    left_go_aa  = m2b_pred_vec[i:i+3];     i += 3
    left_hp_aa  = m2b_pred_vec[i:i+45];    i += 45  # 15*3
    right_go_aa = m2b_pred_vec[i:i+3];     i += 3
    right_hp_aa = m2b_pred_vec[i:i+45];    i += 45
    betas       = m2b_pred_vec[i:i+20];    i += 20
    cam         = m2b_pred_vec[i:i+6];     i += 6

    left_hand = {
        "is_right": False,
        "global_orient": aa_to_R_list(left_go_aa, 1),
        "hand_pose":     aa_to_R_list(left_hp_aa, 15),
        "betas": betas[:10].tolist(),
        "cam":   cam[:3].tolist()
    }
    right_hand = {
        "is_right": True,
        "global_orient": aa_to_R_list(right_go_aa, 1),
        "hand_pose":     aa_to_R_list(right_hp_aa, 15),
        "betas": betas[10:].tolist(),
        "cam":   cam[3:].tolist()
    }
    pred_json = {"persons": {"person_0": left_hand, "person_1": right_hand}}

    # 保存
    save_path = Path(img_path).with_suffix("").as_posix() + "_infer.json"
    with open(save_path, "w") as f:
        json.dump(pred_json, f, indent=2)
    print(f"[OK] Saved prediction to {save_path}")

    return pred_json, save_path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--img_path', type=str, required=True)
    ap.add_argument('--ckp_path', type=str, required=True)
    ap.add_argument('--data_root', type=str, required=True)
    ap.add_argument('--json_path', type=str, required=True)
    ap.add_argument('--hidden_dim', type=int, default=768)
    ap.add_argument('--num_heads', type=int, default=12)
    ap.add_argument('--mask_ratio', type=float, default=0.7)
    ap.add_argument('--no_amp', action='store_true')
    args = ap.parse_args()

    infer_one(
        img_path=args.img_path,
        ckp_path=args.ckp_path,
        data_root=args.data_root,
        json_path=args.json_path,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        mask_ratio=args.mask_ratio,
        use_amp=(not args.no_amp)
    )

if __name__ == '__main__':
    main()
