import math,torch
from pafr_vq.losses.rate import combination_bits,bitrate_metrics
def test_lgamma_combination_edges_and_small_exact_values():
 assert combination_bits(8,0).item()==0 and combination_bits(8,8).item()==0;assert abs(combination_bits(5,2).item()-math.log2(10))<1e-8
 ids=torch.zeros(2,2,2,dtype=torch.long);mask=torch.zeros_like(ids,dtype=torch.bool);assert torch.isfinite(bitrate_metrics(ids,mask,16,8)["total_bits_per_image"])
