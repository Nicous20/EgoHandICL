import torch
import torch.nn as nn
from hamer.models.backbones.vit import ViT
import numpy as np

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch

# Note: The default behavior now has injection attack prevention off.
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-7B", trust_remote_code=True)
tokenizer.eos_token = "<|endoftext|>"
tokenizer.pad_token = tokenizer.eos_token

# use bf16
text_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-7B", device_map="auto", trust_remote_code=True, bf16=True).eval()


class TextCompressor(nn.Module):
    def __init__(self, input_dim, output_dim, num_slots=4):
        super().__init__()
        self.num_slots = num_slots
        self.attn = nn.MultiheadAttention(embed_dim=input_dim, num_heads=8, batch_first=True)
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, text_feat):
        B, T, D = text_feat.shape

        queries = torch.zeros(B, self.num_slots, D, device=text_feat.device, dtype=text_feat.dtype)
        queries = queries.to(torch.float32)
        text_feat = text_feat.to(torch.float32)
        attended, _ = self.attn(queries, text_feat, text_feat)  
        return self.linear(attended)        


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, context):
        B, N, C = x.shape
        q = self.q_proj(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k_proj(context).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v_proj(context).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.out_proj(x)
        return x

class HandICL(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.hidden_dim = config['hidden_dim']
        self.num_heads = config['num_heads']
        self.mano_dim = 157 * 2  
        self.num_pairs = 4 

        self.vit = ViT(
            img_size=224,
            patch_size=16,
            in_chans=3,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4,
            drop_rate=0.0,
            drop_path_rate=0.0
        ).to(self.device)

        self.img_proj = nn.Linear(768, self.hidden_dim)

        self.mano_encoder = nn.Sequential(
            nn.Linear(self.mano_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        

        self.cross_attn1 = CrossAttention(self.hidden_dim, self.num_heads)
        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.hidden_dim,
            nhead=self.num_heads,
            dim_feedforward=4*self.hidden_dim,
            dropout=0.1
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=4
        )

        self.text_compressor = TextCompressor(input_dim=4096, output_dim=self.hidden_dim, num_slots=4)
        self.fusion = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim)
        )


        self.cross_attn2 = CrossAttention(self.hidden_dim, self.num_heads)

        self.predictor = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.mano_dim)
        )
    
    def forward(self, batch):
        B = batch['img1'].shape[0]

        with torch.no_grad():
            img1_feat = self.vit(batch['img1'].to(self.device))  # [B, 768, 14, 14]
            img2_feat = self.vit(batch['img2'].to(self.device))  # [B, 768, 14, 14]

        img1_feat = img1_feat.permute(0, 2, 3, 1)  # [B, 14, 14, 768]
        img2_feat = img2_feat.permute(0, 2, 3, 1)  # [B, 14, 14, 768]
        img1_feat = img1_feat.reshape(B, 196, 768)  # [B, 196, 768]
        img2_feat = img2_feat.reshape(B, 196, 768)  # [B, 196, 768]

        img1_feat = self.img_proj(img1_feat)  # [B, 196, hidden_dim]
        img2_feat = self.img_proj(img2_feat)  # [B, 196, hidden_dim]
        with torch.no_grad():
            text1 = batch['text1']
            text2 = batch['text2']
            inputs1 = tokenizer(
                text1,
                return_tensors='pt',
                padding=True,
                truncation=True
            ).to(text_model.device)
            outputs1 = text_model(**inputs1, output_hidden_states=True)
            hidden_states1 = outputs1.hidden_states
            text_last_hidden1 = hidden_states1[-1]
            
            inputs2 = tokenizer(
                text2,
                return_tensors='pt',
                padding=True,
                truncation=True
            ).to(text_model.device)
            outputs2 = text_model(**inputs2, output_hidden_states=True)
            hidden_states2 = outputs2.hidden_states
            text_last_hidden2 = hidden_states2[-1]
            
        compressed_text1 = self.text_compressor(text_last_hidden1)  # [B, 4, hidden_dim]
        compressed_text2 = self.text_compressor(text_last_hidden2)  # [B, 4, hidden_dim]


        mano_params = []
        for key in ['m1a', 'm1b', 'm2a', 'm2b']:
            param = batch['mano_params'][key].to(self.device)
            param = param.reshape(B, -1)  # [B, 314]
            mano_params.append(param)

        encoded_params = []
        for param in mano_params:
            encoded = self.mano_encoder(param)  # [B, hidden_dim]
            encoded_params.append(encoded.unsqueeze(1))  # [B, 1, hidden_dim]

        mano_feat = torch.cat(encoded_params, dim=1)  # [B, 4, hidden_dim]

        attn_feat = self.cross_attn1(mano_feat, img1_feat)  # [B, 4, hidden_dim]
        fusion_feat = self.fusion(torch.cat([attn_feat, compressed_text1], dim=-1)) # [B, 4, hidden_dim]
        
        # Transformer Decoder
        fusion_feat = fusion_feat.transpose(0, 1)  # [4, B, hidden_dim]
        img1_feat = img1_feat.transpose(0, 1)  # [196, B, hidden_dim]
        decoded_feat = self.transformer_decoder(fusion_feat, img1_feat)  # [4, B, hidden_dim]
        decoded_feat = decoded_feat.transpose(0, 1)  # [B, 4, hidden_dim]

        final_feat = self.cross_attn2(decoded_feat, img2_feat)  # [B, 4, hidden_dim]
        final_feat = self.fusion(torch.cat([final_feat, compressed_text2], dim=-1))  # [B, 4, hidden_dim]


        predictions = []
        for i in range(4):
            pred = self.predictor(final_feat[:, i])  # [B, mano_dim]
            predictions.append(pred)

        output = torch.cat(predictions, dim=1)  # [B, mano_dim*4]
        return output

from handicl.mano_param_utils import decode_mano_params_torch
import warnings
warnings.filterwarnings("ignore", message="You are using a MANO model, with only 10 shape coefficients.")
from hamer.models.mano_wrapper import MANO
from hamer.utils.geometry import perspective_projection

# result = decode_mano_params(encoded_params)


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

def torch_mpjpe(predicted, target):
    # predicted, target: [B, J, 3] or [J, 3]
    return torch.norm(predicted - target, dim=-1).mean()


def torch_p_mpjpe(predicted, target):
    """
    Pose error: MPJPE after rigid alignment (scale, rotation, and translation),
    often referred to as "Protocol #2" in many papers.
    """
    assert predicted.shape == target.shape


    muX = torch.mean(target, dim=1, keepdim=True)
    muY = torch.mean(predicted, dim=1, keepdim=True)
    X0 = target - muX
    Y0 = predicted - muY
    normX = torch.sqrt(torch.sum(X0 ** 2, dim=(1, 2), keepdim=True))
    normY = torch.sqrt(torch.sum(Y0 ** 2, dim=(1, 2), keepdim=True))
    X0 = X0 / normX
    Y0 = Y0 / normY
    H = torch.matmul(X0.permute(0, 2, 1), Y0)
    U, s, Vh = torch.linalg.svd(H)
    V = Vh.transpose(1, 2)
    R = torch.matmul(V, U.transpose(1, 2))
    detR = torch.linalg.det(R)
    sign_detR = torch.sign(detR)

    V = V.clone()
    s = s.clone()
    V = torch.cat([V[:, :, :-1], (V[:, :, -1] * sign_detR.unsqueeze(1)).unsqueeze(2)], dim=2)
    s = torch.cat([s[:, :-1], (s[:, -1] * sign_detR).unsqueeze(1)], dim=1)
    R = torch.matmul(V, U.transpose(1, 2))
    tr = torch.sum(s, dim=1, keepdim=True).unsqueeze(2)
    a = tr * normX / normY
    t = muX - a * torch.matmul(muY, R)
    predicted_aligned = a * torch.matmul(predicted, R) + t
    return torch.mean(torch.norm(predicted_aligned - target, dim=-1))

def compute_mano_output(icl_dict, device):
    mano = MANO(

        create_body_pose=False
    ).to(device)

    pred_mano_params = {}
    # Helper to robustly convert to tensor on device, preserving grad if already tensor
    def robust_tensor(x):
        if torch.is_tensor(x):
            return x.to(device=device, dtype=torch.float32)
        return torch.as_tensor(x, dtype=torch.float32, device=device)

    pred_mano_params['global_orient'] = torch.stack([
        robust_tensor(icl_dict['persons']['person_0']['global_orient']),
        robust_tensor(icl_dict['persons']['person_1']['global_orient'])
    ], dim=0)
    pred_mano_params['hand_pose'] = torch.stack([
        robust_tensor(icl_dict['persons']['person_0']['hand_pose']),
        robust_tensor(icl_dict['persons']['person_1']['hand_pose'])
    ], dim=0)
    pred_mano_params['betas'] = torch.stack([
        robust_tensor(icl_dict['persons']['person_0']['betas']),
        robust_tensor(icl_dict['persons']['person_1']['betas'])
    ], dim=0)

    pred_cam = torch.stack([
        robust_tensor(icl_dict['persons']['person_0']['cam']),
        robust_tensor(icl_dict['persons']['person_1']['cam'])
    ], dim=0)
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

    
from handicl.mano_param_utils import decode_mano_params_torch

import sys

from handicl.model.Uni3D.models import uni3d as models
from handicl.model.Uni3D.utils.params import parse_args


def preprocess_mano_points(pc_xyz):
    """

    """
    if isinstance(pc_xyz, np.ndarray):
        pc_xyz = torch.from_numpy(pc_xyz).float()
    dummy_rgb = torch.ones_like(pc_xyz)  
    pc = torch.cat([pc_xyz, dummy_rgb], dim=-1)
    return pc.unsqueeze(0)


args, _ = parse_args([
    "--model", "create_uni3d",
    "--pc-model", "eva02_tiny_patch14_224",
    "--pc-feat-dim", "192",
    "--embed-dim", "1024",
    "--pc-encoder-dim", "512",
    "--group-size", "64",
    "--num-group", "512",
])
model = models.create_uni3d(args)
ckpt = torch.load("/home/shiqiu/binzhu/Point-In-Context/hand_model/hamer/handicl/model/Uni3D/checkpoints/model.pt", map_location="cpu")
sd = ckpt["module"] if "module" in ckpt else ckpt["state_dict"]
model.load_state_dict(sd, strict=False)
model = model.eval().cuda()
for p in model.parameters():
    p.requires_grad_(False) 

def calc_3d_loss(pred_mano_output, target_mano_output, device):
    pred_hands = pred_mano_output['pred_vertices'].reshape(-1, 1556, 3)[0]
    pred_L, pred_R = pred_hands[:778, :], pred_hands[778:, :]
    tgt_hands = target_mano_output['pred_vertices'].reshape(-1, 1556, 3)[0]
    tgt_L, tgt_R = tgt_hands[:778, :], tgt_hands[778:, :]

    pred_L_pc = preprocess_mano_points(pred_L).to(device=device)   # [1, 778, 6]
    pred_R_pc = preprocess_mano_points(pred_R).to(device=device)
    tgt_L_pc  = preprocess_mano_points(tgt_L).to(device=device)
    tgt_R_pc  = preprocess_mano_points(tgt_R).to(device=device)

    import pdb; pdb.set_trace()
    emb_pred_L = model.encode_pc(pred_L_pc)         # [1, D], requires_grad=True
    emb_pred_R = model.encode_pc(pred_R_pc)


    with torch.no_grad():
        emb_tgt_L = model.encode_pc(tgt_L_pc)
        emb_tgt_R = model.encode_pc(tgt_R_pc)
    emb_tgt_L = emb_tgt_L.detach()
    emb_tgt_R = emb_tgt_R.detach()


    import torch.nn.functional as F


    cos_L = F.cosine_similarity(emb_pred_L, emb_tgt_L, dim=1).mean()
    cos_R = F.cosine_similarity(emb_pred_R, emb_tgt_R, dim=1).mean()
    cos_dist = (2.0 - (cos_L + cos_R)) / 2.0  


    euc_L = torch.norm(emb_pred_L - emb_tgt_L, p=2, dim=1).mean()
    euc_R = torch.norm(emb_pred_R - emb_tgt_R, p=2, dim=1).mean()
    euc_avg = 0.5 * (euc_L + euc_R)

    emb_3d_loss = euc_avg + cos_dist

    return emb_3d_loss

def mano_loss(pred, target):
    batch_size = pred.shape[0]
    total_mpjpe_loss = None
    total_pmpjpe_loss = None
    total_mpvpe = None
    total_3d_loss = None
    for i in range(batch_size):
        pred_dict = decode_mano_params_torch(pred[i])
        pred_m1b_mano_output = compute_mano_output(pred_dict['m1b'], device=pred.device)
        pred_m2b_mano_output = compute_mano_output(pred_dict['m2b'], device=pred.device)

        target_dict = decode_mano_params_torch(target[i])
        target_m1b_mano_output = compute_mano_output(target_dict['m1b'], device=target.device)
        target_m2b_mano_output = compute_mano_output(target_dict['m2b'], device=target.device)

        pred_m1b_mano_output['pred_vertices'] -= pred_m1b_mano_output['pred_vertices'][:, :1, :]
        pred_m2b_mano_output['pred_vertices'] -= pred_m2b_mano_output['pred_vertices'][:, :1, :]
        target_m1b_mano_output['pred_vertices'] -= target_m1b_mano_output['pred_vertices'][:, :1, :]
        target_m2b_mano_output['pred_vertices'] -= target_m2b_mano_output['pred_vertices'][:, :1, :]    

        mpvpe_m1b = torch_mpvpe(pred_m1b_mano_output['pred_vertices'].reshape(1, -1, 3), target_m1b_mano_output['pred_vertices'].reshape(1, -1, 3))
        mpvpe_m2b = torch_mpvpe(pred_m2b_mano_output['pred_vertices'].reshape(1, -1, 3), target_m2b_mano_output['pred_vertices'].reshape(1, -1, 3))
        mpvpe = (mpvpe_m1b + mpvpe_m2b) / 2 
        if total_mpvpe is None:
            total_mpvpe = mpvpe
        else:
            total_mpvpe = total_mpvpe + mpvpe  # out-of-place
        
        #     mpjpe = torch_mpjpe(pred_mano_output['pred_keypoints_3d'], target_mano_output['pred_keypoints_3d'])
        #     pmpjpe = torch_p_mpjpe(pred_mano_output['pred_keypoints_3d'], target_mano_output['pred_keypoints_3d'])
        #     if total_mpjpe_loss is None:
        #         total_mpjpe_loss = mpjpe
        #     else:
        #         total_mpjpe_loss = total_mpjpe_loss + mpjpe  # out-of-place
        #     if total_pmpjpe_loss is None:
        #         total_pmpjpe_loss = pmpjpe
        #     else:
        #         total_pmpjpe_loss = total_pmpjpe_loss + pmpjpe  # out-of-place
        # total_mpjpe_loss = total_mpjpe_loss / batch_size
        # total_pmpjpe_loss = total_pmpjpe_loss / batch_size
        

        # -------- 3d embedding loss -------- #
        emb_3d_loss_m1 = calc_3d_loss(pred_m1b_mano_output, target_m1b_mano_output, pred.device)
        emb_3d_loss_m2 = calc_3d_loss(pred_m2b_mano_output, target_m2b_mano_output, pred.device)
        
        emb_3d_loss = (emb_3d_loss_m1 + emb_3d_loss_m2) / 2

        if total_3d_loss is None:
            total_3d_loss = emb_3d_loss
        else:
            total_3d_loss = total_3d_loss + emb_3d_loss
        # -------- 3d embedding loss -------- #

    total_mpvpe = total_mpvpe / batch_size
    total_3d_loss = total_3d_loss / batch_size
    total_loss = total_mpvpe + total_3d_loss * 1e-3
    # total_loss = total_mpvpe

    return total_loss