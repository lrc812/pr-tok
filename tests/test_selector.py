import torch
from pafr_vq.allocation import BudgetSelector
def test_selector_exact_budget_and_seed():
 score=torch.zeros(3,4,4);a=BudgetSelector(deterministic=True,seed=3);m1,i1=a(score,budget=5,random=True);m2,i2=a(score,budget=5,random=True);assert torch.equal(m1,m2) and torch.equal(i1,i2) and torch.equal(m1.sum((1,2)),torch.full((3,),5))
