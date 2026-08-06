import torch

def custom_collate_fn(batch):
    imgs, t5_feats, attn_masks, valids = zip(*batch)
    imgs = torch.stack(imgs)  # 拼接图像
    t5_feats = torch.cat(t5_feats, dim=0)  # 拼接 T5 特征
    attn_masks = torch.cat(attn_masks, dim=0)  # 拼接注意力掩码
    valids = torch.cat(valids)  # 拼接 valid 标志
    return imgs, t5_feats, attn_masks, valids
