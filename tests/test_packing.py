import torch
from pafr_vq.utils.packing import pack_active_tokens,unpack_active_tokens
def test_pack_unpack_round_trip():
 ids=torch.tensor([[[1,9,2],[9,3,9]],[[4,5,9],[9,9,6]]]);mask=ids!=9;packed=pack_active_tokens(ids,mask);assert torch.equal(unpack_active_tokens(packed,mask,9),ids);assert torch.equal(packed.positions[0,:3],torch.tensor([0,2,4]))
