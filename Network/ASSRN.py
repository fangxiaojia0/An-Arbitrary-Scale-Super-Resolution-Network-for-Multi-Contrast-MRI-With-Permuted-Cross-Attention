import torch
import torch.nn as nn
import torch.nn.functional as F
import models
from models import register
from models.CFE import cfe_blcok
from models.RegModu import Reg
from utils import make_coord, CrossScaleAttention, Transformer2D


@register('assrn')
class ASSRN(nn.Module):

    def __init__(self, 
                 encoder_spec, 
                 imnet_q_spec,
                 imnet_k_spec,
                 imnet_v_spec,
                 imnet_r_spec,
                 imnet_spec=None,
                 local_size=2,
                 feat_unfold=True,
                 non_local_attn=False,
                 multi_scale=[2],
                 softmax_scale=1,                  
                 hidden_dim=256):
        super().__init__()

        self.feat_unfold = feat_unfold
        self.local_size = local_size
        self.non_local_attn = non_local_attn
        self.multi_scale = multi_scale
        self.softmax_scale = softmax_scale
        self.encoder = models.make(encoder_spec)
        imnet_dim = self.encoder.out_dim
        if self.feat_unfold:
            imnet_q_spec['args']['in_dim'] = imnet_dim * 9 * 4
            imnet_k_spec['args']['in_dim'] = imnet_k_spec['args']['out_dim'] = imnet_dim * 9
            imnet_v_spec['args']['in_dim'] = imnet_v_spec['args']['out_dim'] = imnet_dim * 9
            imnet_r_spec['args']['in_dim'] = imnet_r_spec['args']['out_dim'] = imnet_dim * 9
        else:
            imnet_q_spec['args']['in_dim'] = imnet_dim
            imnet_k_spec['args']['in_dim'] = imnet_k_spec['args']['out_dim'] = imnet_dim
            imnet_v_spec['args']['in_dim'] = imnet_v_spec['args']['out_dim'] = imnet_dim
            imnet_r_spec['args']['in_dim'] = imnet_r_spec['args']['out_dim'] = imnet_dim

        imnet_k_spec['args']['in_dim'] += 4
        imnet_v_spec['args']['in_dim'] += 4
        imnet_r_spec['args']['in_dim'] += 4

        if self.non_local_attn:
            imnet_q_spec['args']['in_dim'] += imnet_dim*len(multi_scale)
            imnet_v_spec['args']['in_dim'] += imnet_dim*len(multi_scale)
            imnet_v_spec['args']['out_dim'] += imnet_dim*len(multi_scale)        

        self.imnet_q = models.make(imnet_q_spec)
        self.imnet_k = models.make(imnet_k_spec)
        self.imnet_v = models.make(imnet_v_spec)
        self.imnet_r = models.make(imnet_r_spec)

        if self.non_local_attn:
            self.cs_attn = CrossScaleAttention(channel=imnet_dim, scale=multi_scale)

        self.net_R = Reg()
        self.trans = Transformer2D()
        self.coef = nn.Conv2d(self.encoder.out_dim, hidden_dim, 3, padding=1)
        self.freq = nn.Conv2d(self.encoder.out_dim, hidden_dim, 3, padding=1)
        self.ref_conv = nn.Conv2d(self.encoder.out_dim, imnet_dim * 10, 3, padding=1)
        self.phase = nn.Linear(2, hidden_dim // 2, bias=False)
        self.local_enhanced_block = cfe_blcok()
        self.imnet = models.make(imnet_spec)


    def gen_feat(self, inp, ref, ref_hr):
        self.inp_reg = F.interpolate(inp, (ref_hr.shape[-2], ref_hr.shape[-1]), mode='bicubic')
        self.inp = inp
        self.ref = ref
        self.ref_hr = ref_hr

        self.flow = self.net_R(self.inp_reg, self.ref_hr)
        self.T1_warp = self.trans(self.inp_reg, self.flow)
        self.flow_ = self.net_R(self.ref_hr, self.inp_reg)
        self.T2_warp = self.trans(self.ref_hr, self.flow_)
        ref_align = self.T2_warp

        self.inp_SRnet = F.interpolate(ref_align, (self.inp.shape[-1], self.inp.shape[-2]), mode='bicubic')
        self.inp_feat_lst = self.encoder(inp, self.inp_SRnet)
        self.ref_feat_hr, self.ref_loss =  self.local_enhanced_block(ref_align)
        B, C, H, W = self.ref_feat_hr.shape
        self.ref_feat_hr_res = F.unfold(self.ref_feat_hr, 3, padding=1).view(B, C*9, H, W)

        return self.inp_feat_lst, self.ref_feat_hr_res, self.ref_loss, self.flow, self.flow_, self.inp_reg, self.T1_warp, self.T2_warp


    def pxiel_query(self, features, ref_feat_hr_res, coord, scale):
        """
        Args:
            feature (Tensor): encoded feature.
            coord (Tensor): coord tensor, shape (BHW, 2).

        Returns:
            result (Tensor): (part of) output.
        """
        for feature in features:  
            B, C, H, W = feature.shape

            if self.feat_unfold:
                feat_q = F.unfold(feature, 3, padding=1).view(B, C*9, H, W)
                feat_k = F.unfold(feature, 3, padding=1).view(B, C*9, H, W)
                if self.non_local_attn:
                    non_local_feat_v = self.cs_attn(feature)
                    feat_v = F.unfold(feature, 3, padding=1).view(B, C*9, H, W) 
                    feat_v = torch.cat([feat_v, non_local_feat_v], dim=1)
                else:
                    feat_v = F.unfold(feature, 3, padding=1).view(B, C*9, H, W)     
            else:
                feat_q = feat_k = feat_v = feature

            # query
            query = F.grid_sample(feat_q, coord.flip(-1).unsqueeze(1), mode='nearest',
                        align_corners=False).permute(0, 3, 2, 1).contiguous()       

            feat_coord = make_coord(feature.shape[-2:], flatten=False).permute(2, 0, 1) \
                            .unsqueeze(0).expand(B, 2, *feature.shape[-2:])
            feat_coord = feat_coord.to(coord) 

            if self.local_size == 1:
                v_lst = [(0, 0)]
            else:
                v_lst = [(i,j) for i in range(-1, 2, 4-self.local_size) for j in range(-1, 2, 4-self.local_size)]
            eps_shift = 1e-6
            pred_x = []
            
            for v in v_lst:
                vx, vy = v[0], v[1]
                # project to LR field
                tx = ((H - 1) / (1 - scale[:,0,0])).view(B,  1)  
                ty = ((W - 1) / (1 - scale[:,0,1])).view(B,  1)   
                rx = (2*abs(vx) -1) / tx if vx != 0 else 0         
                ry = (2*abs(vy) -1) / ty if vy != 0 else 0         
                bs, q = coord.shape[:2]     
                coord_ = coord.clone()
                if vx != 0:
                    coord_[:, :, 0] += vx /abs(vx) * rx + eps_shift
                if vy != 0:
                    coord_[:, :, 1] += vy /abs(vy) * ry + eps_shift  
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)

                # key and value
                key = F.grid_sample(feat_k, coord_.flip(-1).unsqueeze(1), mode='nearest', 
                    align_corners=False)[:, :, 0, :].permute(0, 2, 1).contiguous()
                value = F.grid_sample(feat_v, coord_.flip(-1).unsqueeze(1), mode='nearest', 
                    align_corners=False)[:, :, 0, :].permute(0, 2, 1).contiguous()
                ref_hr_resp = F.grid_sample(ref_feat_hr_res, coord_.flip(-1).unsqueeze(1),
                    align_corners=False)[:, :, 0, :].permute(0, 2, 1).contiguous()

                coord_k = F.grid_sample(feat_coord, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :].permute(0, 2, 1)

                Q, K = coord, coord_k
                rel = Q - K
                rel[:, :, 0] *= feature.shape[-2] 
                rel[:, :, 1] *= feature.shape[-1]  
                inp = rel  

                scale_ = scale.clone()
                scale_[:, :, 0] *= feature.shape[-2]
                scale_[:, :, 1] *= feature.shape[-1]

                inp_v = torch.cat([value, inp, scale_], dim=-1)
                inp_k = torch.cat([key, inp, scale_], dim=-1)
                inp_r = torch.cat([ref_hr_resp, inp, scale_], dim=-1)    

                inp_k = inp_k.contiguous().view(bs * q, -1) 
                inp_v = inp_v.contiguous().view(bs * q, -1) 
                inp_r = inp_r.contiguous().view(bs * q, -1) 

                weight_k = self.imnet_k(inp_k).view(bs, q, -1).contiguous()
                pred_k = (key * weight_k).view(bs, q, -1)    
                
                weight_v = self.imnet_v(inp_v).view(bs, q, -1).contiguous()
                pred_v = (value * weight_v).view(bs, q, -1)                 

                weight_r = self.imnet_r(inp_r).view(bs, q, -1).contiguous()
                pred_r = (ref_hr_resp * weight_r).view(bs, q, -1)            

                attn = (query @ pred_k.unsqueeze(-1))
                x = (((attn/self.softmax_scale).softmax(dim=-1) @ pred_v.unsqueeze(2)) + pred_r.unsqueeze(2)).view(bs*q, -1)

                pred_x.append(x)

        result = torch.cat(pred_x, dim=-1)
        result = self.imnet_q(result).view(bs, q, -1)

        return result


    def forward(self, x, coord, cell, ref, ref_hr, test_mode=False):
        """Forward function.

        Args:
            x: input tensor.
            coord (Tensor): coordinates tensor.
            cell (Tensor): cell tensor.
            ref (Tensor): ref tensor.
            ref_hr (Tensor): ref_hr tensor.
            test_mode (bool): Whether in test mode or not. Default: False.

        Returns:
            pred (Tensor): output of model.
        """

        feature, ref_feat_hr_res, ref_loss, flow, flow_, inp_reg, T1_warp, T2_warp = self.gen_feat(x, ref, ref_hr)
        pred = self.pxiel_query(feature, ref_feat_hr_res, coord, cell)
        pred += F.grid_sample(x, coord.flip(-1).unsqueeze(1), mode='bilinear',\
                    padding_mode='border', align_corners=False)[:, :, 0, :].permute(0, 2, 1)

        return pred, ref_loss, flow, flow_, inp_reg, T1_warp, T2_warp
