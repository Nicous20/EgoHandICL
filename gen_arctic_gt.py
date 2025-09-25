import numpy as np
import json
import os
from pathlib import Path
from tqdm import tqdm


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

def fix_global_orient(global_orient):
    """

    """
    R_r = np.array([[-1, 0, 0],
                       [0, 1, 0],
                       [0, 0, -1]])
    
    R = np.array(global_orient)
    if R.shape == (1, 3, 3):
        R = R[0]
    R_fixed = R_r @ R
    return R_fixed.reshape(1, 3, 3).tolist()


def convert_arctic_to_gt(npy_path, save_dir):

    npy = np.load(npy_path, allow_pickle=True).item()
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    img_dir = npy_path.split('/')[-1].split('.')[0]
    mano_mean_left_pose = np.load('')
    mano_mean_right_pose = np.load('')
    
    # import pdb; pdb.set_trace()


    for idx in tqdm(range(len(npy['left']['pose']))):
        # import pdb; pdb.set_trace() 
        mano_gt = {
            "persons": {
                "person_0": {
                    "is_right": False,
                    "hand_pose": [],
                    "global_orient": [],
                    "betas": [],
                    "cam": []
                },
                "person_1": {
                    "is_right": True,
                    "hand_pose": [],
                    "global_orient": [],
                    "betas": [],
                    "cam": []
                }
            }
        }
        import pdb; pdb.set_trace()

        rot_matrix = np.array(npy['left']['rot'][idx])  # 1 * 3

        from scipy.spatial.transform import Rotation as R
        rot_matrix = R.from_rotvec(rot_matrix).as_matrix()

        mano_gt['persons']['person_0']['global_orient'] = rot_matrix.reshape(1, 3, 3).tolist()

        
        pose_split = np.split(npy['left']['pose'][idx] + mano_mean_left_pose, 15)
        for i in pose_split:
            mano_gt['persons']['person_0']['hand_pose'].append(R.from_rotvec(i).as_matrix().tolist())

        mano_gt['persons']['person_0']['betas'] = npy['left']['shape'].tolist()
        mano_gt['persons']['person_0']['cam'] = np.array(npy['left']['trans'][idx]).tolist()
        # import pdb; pdb.set_trace()
        # mirror to right
        # mano_gt['persons']['person_0'] = mirror_left_to_right_mano(mano_gt['persons']['person_0'])


        rot_matrix = np.array(npy['right']['rot'][idx])  # 1 * 3
        rot_matrix = R.from_rotvec(rot_matrix).as_matrix()
        mano_gt['persons']['person_1']['global_orient'] = rot_matrix.reshape(1, 3, 3).tolist()

        pose_split = np.split(npy['right']['pose'][idx] + mano_mean_right_pose, 15)
        for i in pose_split:
            mano_gt['persons']['person_1']['hand_pose'].append(R.from_rotvec(i).as_matrix().tolist())

        mano_gt['persons']['person_1']['betas'] = npy['right']['shape'].tolist()
        mano_gt['persons']['person_1']['cam'] = np.array(npy['right']['trans'][idx]).tolist()

        # import pdb; pdb.set_trace()
        # mano_gt['persons']['person_0']['global_orient'] = fix_global_orient(
        #     mano_gt['persons']['person_0']['global_orient'])
        # mano_gt['persons']['person_1']['global_orient'] = fix_global_orient(
        #     mano_gt['persons']['person_1']['global_orient']
        # )

        json_file = img_dir + f'_0_{idx+1:05d}_gt.json'  
        json_file = Path(save_dir) / json_file
        with open(json_file, 'w') as f:
            json.dump(mano_gt, f, indent=2)

if __name__ == '__main__':

    npy_dir = ''
    save_dir = ''
    for npy_path in tqdm(os.listdir(npy_dir)):
        convert_arctic_to_gt(os.path.join(npy_dir, npy_path), save_dir) 