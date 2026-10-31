"""Tiny weighted-loss invariants, independent of training data/model."""
import modal
from student_train import image
app = modal.App('tilawa-targeted-checks', image=image.add_local_python_source('student_train'))
@app.function(cpu=2, memory=2048)
def checks():
    import torch
    import torch.nn.functional as F
    from student import soft_target_loss
    torch.manual_seed(0)
    logits = torch.randn(2, 8, 5, requires_grad=True)
    lp = logits.log_softmax(-1)
    lens = torch.tensor([8,6])
    target_lens = torch.tensor([3,2])
    targets = torch.tensor([1,2,3,2,1])
    raw=F.ctc_loss(lp.transpose(0,1),targets,lens,target_lens,blank=0,reduction='none')
    old=F.ctc_loss(lp.transpose(0,1),targets,lens,target_lens,blank=0,reduction='sum')/lens.sum()
    ones=torch.ones(2)
    new=(raw*ones).sum()/(lens*ones).sum()
    assert torch.allclose(old,new)
    ids=torch.tensor([[[0,1]]*8]*2)
    probs=torch.tensor([[[.7,.3]]*8]*2)
    valid=torch.arange(8)[None,:]<lens[:,None]
    assert torch.equal(soft_target_loss(lp,ids,probs,valid),soft_target_loss(lp,ids,probs,valid.float()*ones[:,None]))
    weights=torch.tensor([3.,1.])
    weighted=(raw*weights).sum()/(lens*weights).sum()
    kd=soft_target_loss(lp,ids,probs,valid.float()*weights[:,None])
    per=[soft_target_loss(lp[i:i+1],ids[i:i+1],probs[i:i+1],valid[i:i+1]) for i in range(2)]
    assert torch.allclose(kd,(per[0]*24+per[1]*6)/30)
    (weighted+kd).backward()
    assert torch.isfinite(logits.grad).all()
    return {'ctcUniformEquivalent':True,'kdUniformEquivalent':True,'manualWeightEquivalent':True,'finiteGradient':True}
@app.local_entrypoint()
def main():
    print(checks.remote())
