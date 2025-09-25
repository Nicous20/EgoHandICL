# handicl/dataset.py
from tkinter import NO
import torch
from torch.utils.data import Dataset
import random
from pathlib import Path
import json
from PIL import Image
import numpy as np
import copy
import cv2



# ... existing code ...
class HandICLDataset(Dataset):
    def __init__(self, data_root, transform=None, mask_ratio=0.7, num_samples=None, json_path=None, type='0'):
        self.data_root = data_root
        self.mask_ratio = mask_ratio
        self.transform = transform
        self.type = type
        import json
        with open(json_path, 'r') as f:
            self.img_dir = json.load(f)

        self.valid_paths = []
        for img_path in self.img_dir:
            gt_path = Path(data_root) / 'mano_gt' / f"{Path(img_path).stem}_gt.json"
            mano_path = Path(data_root) / 'mano_hamer' / f"{Path(img_path).stem}_mano.json"
            if mano_path.exists() and gt_path.exists():
                self.valid_paths.append(img_path)
        

        if num_samples is not None and len(self.valid_paths) > num_samples:
            self.valid_paths = random.sample(self.valid_paths, num_samples)
        
        print(f"Using {len(self.valid_paths)} samples for training")
    
    def random_mask(self, mano_params):

        mask = torch.rand(mano_params.shape) > self.mask_ratio
        return mano_params * mask
    
    def get_mano_pair(self, img_path):


        img_path = Path(img_path)
        try:

            mano_dir = Path(self.data_root) / 'mano_wilor'  
            pred_mano_path = mano_dir / f"{img_path.stem}_wilor.json"

            gt_dir = Path(self.data_root) / 'mano_gt'
            gt_mano_path = gt_dir / f"{img_path.stem}_gt.json"



            with open(pred_mano_path, 'r') as f:
                pred_mano = json.load(f)
            

            with open(gt_mano_path, 'r') as f:
                gt_mano = json.load(f)
            
            return pred_mano, gt_mano
            
        except Exception as e:
            print(f"Error in get_mano_pair for {img_path}")
            print(f"Pred path: {pred_mano_path}")
            print(f"GT path: {gt_mano_path}")
            print(f"Error: {str(e)}")
            raise
    
    def mirror_right_to_left_mano(self, right_mano_dict: dict) -> dict:
        """
        Mirror MANO right-hand parameters (in JSON/dict format) to left-hand version.

        Args:
            right_mano_dict (dict): a right-hand MANO json dict. Format:
                - 'hand_pose': List[15][3][3]
                - 'global_orient': List[1][3][3]
                - 'betas': List[10]
                - 'cam': List[3]

        Returns:
            dict: mirrored left-hand MANO dict in the same format
        """
        def mirror_rotmat_np(rotmat: np.ndarray) -> np.ndarray:
            # rotmat: [..., 3, 3]
            M = np.diag([-1.0, 1.0, 1.0])  # mirror matrix across YZ plane
            return M @ rotmat @ M.T

        # Deep copy the dict
        left_mano_dict = copy.deepcopy(right_mano_dict)

        # Convert and mirror hand_pose
        hand_pose = np.array(right_mano_dict['hand_pose'])  # shape (15, 3, 3)
        hand_pose_mirrored = mirror_rotmat_np(hand_pose)
        left_mano_dict['hand_pose'] = hand_pose_mirrored.tolist()

        # Convert and mirror global_orient
        global_orient = np.array(right_mano_dict['global_orient'])  # shape (1, 3, 3)
        global_orient_mirrored = mirror_rotmat_np(global_orient)
        left_mano_dict['global_orient'] = global_orient_mirrored.tolist()

        # betas and cam remain the same
        left_mano_dict['betas'] = right_mano_dict['betas']
        left_mano_dict['cam'] = right_mano_dict['cam']

        # Update handedness tag if present
        # if 'is_right' in left_mano_dict:
        #     left_mano_dict['is_right'] = False

        return left_mano_dict

    def fill_hand(self, mano_dict, img_path):
        mano_hamer_dir = Path(self.data_root) / 'mano_hamer'
        mano_hamer_path = mano_hamer_dir / f"{Path(img_path).stem}_mano.json"
        with open(mano_hamer_path, 'r') as f:
            mano_hamer = json.load(f)
        mano_left_hand = None
        mano_right_hand = None
        for person_id in mano_hamer['persons']:
            if mano_hamer['persons'][person_id]['is_right']:
                mano_right_hand = mano_hamer['persons'][person_id]
            else:
                mano_left_hand = mano_hamer['persons'][person_id]
        left_hand = None
        right_hand = None
        for person_id in mano_dict['persons']:
            if mano_dict['persons'][person_id]['is_right'] < 1e-5:
                left_hand = mano_dict['persons'][person_id]                
            else:
                right_hand = mano_dict['persons'][person_id]
        if left_hand is None:
            left_hand = mano_left_hand
        if right_hand is None:
            right_hand = mano_right_hand
        mano_filled = {
            'persons': {
                'person_0': left_hand,
                'person_1': right_hand
            }
        }
        return mano_filled 

    
    def process_mano(self, mano_dict):

        left_hand = mano_dict['persons']['person_0']
        right_hand = mano_dict['persons']['person_1']
        left_params = torch.cat([
            torch.tensor([cv2.Rodrigues(np.array(R))[0].flatten().tolist() for R in left_hand['global_orient']]).flatten(),
            torch.tensor([cv2.Rodrigues(np.array(R))[0].flatten().tolist() for R in left_hand['hand_pose']]).flatten(),
            torch.tensor(left_hand['betas']).flatten(),
            torch.tensor(left_hand['cam']).flatten()
        ])
        right_params = torch.cat([
            torch.tensor([cv2.Rodrigues(np.array(R))[0].flatten().tolist() for R in right_hand['global_orient']]).flatten(),
            torch.tensor([cv2.Rodrigues(np.array(R))[0].flatten().tolist() for R in right_hand['hand_pose']]).flatten(),
            torch.tensor(right_hand['betas']).flatten(),
            torch.tensor(right_hand['cam']).flatten()
        ])

        return torch.cat([
            left_params,
            right_params,
        ]).to(torch.float32) 
    
    def __getitem__(self, idx):
        img1_path = self.valid_paths[idx]
        img2_path = random.choice(self.valid_paths)
        while img1_path == img2_path:
            img2_path = random.choice(self.valid_paths)
        
        img1 = self.transform(Image.open(img1_path).convert('RGB'))
        img2 = self.transform(Image.open(img2_path).convert('RGB'))
        

        m1a, m1b = self.get_mano_pair(img1_path)
        
        
        m2a, m2b = self.get_mano_pair(img2_path)
        
        


        m1a = self.fill_hand(m1a, img1_path)
        m2a = self.fill_hand(m2a, img2_path)

        m1a = self.process_mano(m1a)
        m1b = self.process_mano(m1b)
        m2a = self.process_mano(m2a)
        m2b = self.process_mano(m2b)
        
        text_instruction_path = ''
        with open(text_instruction_path, 'r') as f:
            text_instruction = json.load(f)

        return {
            'type': self.type,
            'img1': img1,
            'img2': img2,
            'text1': text_instruction[img1_path.split('/')[-1].split('.')[0]],
            'text2': text_instruction[img2_path.split('/')[-1].split('.')[0]],
            'mano_params': {
                'm1a': m1a,
                'm1b': m1b,
                'm2a': m2a,
                'm2b': m2b
            },
            'target': torch.cat([m1a, m1b, m2a, m2b], dim=0),
            'mask_ratio': self.mask_ratio
        }
            
    
    def __len__(self):
        return len(self.valid_paths)