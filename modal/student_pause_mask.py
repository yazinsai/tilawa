"""Pause-blocked streaming student, shared by training and ONNX deployment.

Cutoffs use the native 20 ms embedding clock. A negative cutoff leaves a query
unchanged. State and positions advance normally; no cache is reset. Each query
can read its own pause warm-up and subsequent frames, but no earlier frame.
"""
from __future__ import annotations

import types
import torch
import torch.nn.functional as F


def allowed_sources(floors, processed, sources, left=0):
    keys = processed[:, None] + torch.arange(sources, device=floors.device)[None] - left
    return (floors[:, :, None] < 0) | (keys[:, None] >= floors[:, :, None])


def time_windows(x,count,width):
    """Indexed sliding windows also export when ONNX Pad loses static sizes."""
    indices=torch.arange(count,device=x.device)[:,None]+torch.arange(width,device=x.device)[None]
    return x.index_select(-1,indices.reshape(-1)).reshape(*x.shape[:-1],count,width)


def stable_activation(self,x):
    """Same eval activation, with an overflow-safe derivative during training.

    Icefall's eval helper selects a finite result after exp overflows, but its
    unused exp branch still poisons backward. Keep inference/export unchanged.
    """
    if torch.jit.is_tracing() or not x.requires_grad:
        return self.pause_original_forward(x)
    if self.activation=='SwooshL': offset,constant=4.,.035
    elif self.activation=='SwooshR': offset,constant=1.,.313261687
    else: raise ValueError(self.activation)
    return F.linear(F.softplus(x-offset)-.08*x-constant,self.weight,self.bias)


def downsample(self,src):
    """A straddling pool cannot lend pre-pause features to later native rows."""
    if not torch.jit.is_tracing() and not self.pause_active:
        return self.pause_original_forward(src)
    floor=self.pause_floor.reshape(src.shape[1],-1,self.downsample).amax(-1)
    floor=floor.repeat_interleave(self.downsample,dim=1)
    keys=self.pause_processed[:,None]+torch.arange(src.shape[0],device=src.device)[None]
    keep=(floor<0) | (keys>=floor)
    return self.pause_original_forward(src*keep.transpose(0,1)[:,:,None])


def attention(self, x, pos_emb, cached_key, left_context_len, key_padding_mask):
    if not torch.jit.is_tracing() and not self.pause_active:
        return self.pause_original(x, pos_emb, cached_key, left_context_len, key_padding_mask)
    projected = self.in_proj(x)
    t, b, _ = projected.shape
    h, d, pd = self.num_heads, self.query_head_dim, self.pos_head_dim
    q, k, p = torch.split(projected, [h*d, h*d, h*pd], dim=-1)
    keys = torch.cat([cached_key, k], dim=0)
    new_key = keys[-left_context_len:]
    n = keys.shape[0]
    q = q.reshape(t,b,h,d).permute(2,1,0,3)
    k = keys.reshape(n,b,h,d).permute(2,1,3,0)
    p = p.reshape(t,b,h,pd).permute(2,1,0,3)
    positions = self.linear_pos(pos_emb).reshape(-1,2*t-1+left_context_len,h,pd).permute(2,0,3,1)
    relative = torch.matmul(p,positions)
    indices = (t-1-torch.arange(t,device=x.device))[:,None] + torch.arange(n,device=x.device)[None]
    relative = torch.gather(relative,-1,indices[None,None].expand(h,b,t,n))
    scores = torch.matmul(q,k) + relative
    if key_padding_mask is not None:
        scores = scores.masked_fill(key_padding_mask.bool()[None,:,None,:],-1000)
    allowed = allowed_sources(self.pause_floor, self.pause_processed,n,left_context_len)
    scores = scores.masked_fill(~allowed[None],-1000)
    # A pool straddling a late cutoff can have no eligible key in this chunk.
    # Its attention contribution is zero, rather than uniform over old keys.
    return scores.softmax(-1).masked_fill(~allowed[None],0), new_key


def depthwise(self, x, cache):
    original, new_cache = self.pause_original(x, cache=cache)
    if not torch.jit.is_tracing() and not self.pause_active:
        return original, new_cache
    b, c, t = x.shape
    left = cache.shape[-1]
    causal = time_windows(torch.cat([cache,x],dim=-1),t,left+1)
    allow = allowed_sources(self.pause_floor,self.pause_processed,t+left,left)
    offsets = torch.arange(t,device=x.device)[:,None]+torch.arange(left+1,device=x.device)[None]
    keep = torch.gather(allow,2,offsets[None].expand(b,-1,-1))
    causal = (causal*keep[:,None]*self.causal_conv.weight[:,0][None,:,None]).sum(-1)
    if self.causal_conv.bias is not None: causal = causal+self.causal_conv.bias[None,:,None]
    width = self.kernel_size
    local = time_windows(F.pad(x,(left,left)),t,width)
    local_keys = self.pause_processed[:,None,None] + torch.arange(t,device=x.device)[None,:,None] + torch.arange(width,device=x.device)[None,None] - left
    keep = (self.pause_floor[:,:,None]<0) | (local_keys>=self.pause_floor[:,:,None])
    local = (local*keep[:,None]*self.chunkwise_conv.weight[:,0][None,:,None]).sum(-1)
    if self.chunkwise_conv.bias is not None: local = local+self.chunkwise_conv.bias[None,:,None]
    masked = causal + local*self._get_chunk_scale(chunk_size=t)
    return torch.where(self.pause_floor[:,None]>=0,masked,original), new_cache


def convnext(self,x,cached_left_pad):
    original,new_pad = self.pause_original(x,cached_left_pad=cached_left_pad)
    if not torch.jit.is_tracing() and not self.pause_active:
        return original, new_pad
    pt,pf = self.padding
    t = x.shape[2]-pt
    source = F.pad(torch.cat([cached_left_pad,x],dim=2),(pf,pf))
    kt,kf = 2*pt+1,2*pf+1
    times=torch.arange(t,device=x.device)[:,None]+torch.arange(kt,device=x.device)[None]
    freqs=torch.arange(x.shape[3],device=x.device)[:,None]+torch.arange(kf,device=x.device)[None]
    windows=source.index_select(2,times.reshape(-1)).reshape(x.shape[0],x.shape[1],t,kt,source.shape[3]).permute(0,1,2,4,3)
    windows=windows.index_select(3,freqs.reshape(-1)).reshape(x.shape[0],x.shape[1],t,x.shape[3],kf,kt).permute(0,1,2,3,5,4)
    keys = self.pause_processed[:,None,None]+torch.arange(t,device=x.device)[None,:,None]+torch.arange(2*pt+1,device=x.device)[None,None]-pt
    keep = (self.pause_floor[:,:t,None]<0) | (keys>=self.pause_floor[:,:t,None])
    hidden = (windows*keep[:,None,:,None,:,None]*self.depthwise_conv.weight[:,0][None,:,None,None]).sum((-1,-2))
    hidden = hidden+self.depthwise_conv.bias[None,:,None,None]
    hidden = self.pointwise_conv1(hidden)
    hidden = self.hidden_balancer(hidden)
    hidden = self.activation(hidden)
    hidden = self.pointwise_conv2(hidden)
    hidden = x[:,:,:t]+hidden
    return torch.where(self.pause_floor[:,None,:t,None]>=0,hidden,original),new_pad


class PauseStreaming(torch.nn.Module):
    def __init__(self,model,chunk=16,left=128):
        super().__init__()
        model.eval()
        model.streaming_eval(chunk,left)
        self.encoder,self.encoder_embed,self.ctc_output = model.encoder,model.encoder_embed,model.ctc_output
        self.chunk,self.left = chunk,left
        from scaling import ActivationDropoutAndLinear
        for module in self.modules():
            if isinstance(module,ActivationDropoutAndLinear):
                module.pause_original_forward=module.forward
                module.forward=types.MethodType(stable_activation,module)
        self.masked = []
        self.pools = []
        for stage,ds in zip(self.encoder.encoders,self.encoder.downsampling_factor):
            if hasattr(stage,'downsample'):
                pool=stage.downsample
                pool.pause_original_forward=pool.forward
                pool.pause_active=True
                pool.forward=types.MethodType(downsample,pool)
                self.pools.append(pool)
            inner = stage.encoder if hasattr(stage,'encoder') else stage
            for layer in inner.layers:
                a = layer.self_attn_weights
                a.pause_original = a.streaming_forward
                a.pause_active = True
                a.streaming_forward = types.MethodType(attention,a)
                self.masked.append((a,ds))
                for conv in (layer.conv_module1.depthwise_conv,layer.conv_module2.depthwise_conv):
                    conv.pause_original = conv.streaming_forward
                    conv.pause_active = True
                    conv.streaming_forward = types.MethodType(depthwise,conv)
                    self.masked.append((conv,ds))
        embed = self.encoder_embed.convnext
        embed.pause_original = embed.streaming_forward
        embed.pause_active = True
        embed.streaming_forward = types.MethodType(convnext,embed)

    def init_states(self,batch=1,device=None):
        device = device or next(self.parameters()).device
        return self.encoder.get_init_states(batch,device)+[self.encoder_embed.get_init_states(batch,device),torch.zeros(batch,dtype=torch.int64,device=device)]

    def forward(self,x,states,pause_floor):
        processed = states[-1]
        active = True if torch.jit.is_tracing() else bool((pause_floor >= 0).any())
        clocks = {}
        for module,ds in self.masked:
            if ds not in clocks:
                # Every native query sharing a pooled frame must respect its
                # cutoff, including an onset inside the pooling group.
                floor = pause_floor.reshape(x.shape[0],-1,ds).amax(-1)
                clocks[ds] = (torch.where(floor<0,floor,(floor+ds-1)//ds),processed//ds)
            module.pause_floor,module.pause_processed = clocks[ds]
            module.pause_active = active
        for pool in self.pools:
            pool.pause_floor,pool.pause_processed=pause_floor,processed
            pool.pause_active=active
        self.encoder_embed.convnext.pause_floor = pause_floor
        self.encoder_embed.convnext.pause_processed = processed
        self.encoder_embed.convnext.pause_active = active
        lengths = torch.full((x.shape[0],),2*self.chunk+13,dtype=torch.int64,device=x.device)
        embedded,lengths,pad = self.encoder_embed.streaming_forward(x,lengths,states[-2])
        mask = (processed[:,None]<=torch.arange(self.left,device=x.device)[None]).int().flip(1)
        mask = torch.cat([mask,torch.zeros(x.shape[0],self.chunk,dtype=torch.int32,device=x.device)],dim=1)
        output,_,new_states = self.encoder.streaming_forward(embedded.permute(1,0,2),lengths,states[:-2],mask)
        return self.ctc_output(output.permute(1,0,2)).log_softmax(-1),new_states+[pad,processed+lengths]


def stream(model,feats,floors=None,detach_every=8):
    """Carried deployed caches, with truncated backpropagation through state only."""
    states = model.init_states(feats.shape[0],feats.device)
    total = ((feats.shape[1]-7)//2+1)//2
    windows = (total+model.chunk//2-1)//(model.chunk//2)
    needed = (windows-1)*2*model.chunk+2*model.chunk+13
    feats = F.pad(feats,(0,0,0,max(0,needed-feats.shape[1])),value=-23.0259)
    rows = []
    for i in range(windows):
        cut = torch.full((feats.shape[0],model.chunk),-1,dtype=torch.int64,device=feats.device) if floors is None else floors[:,i*model.chunk:(i+1)*model.chunk]
        if cut.shape[1]<model.chunk: cut = F.pad(cut,(0,model.chunk-cut.shape[1]),value=-1)
        lp,states = model(feats[:,i*2*model.chunk:i*2*model.chunk+2*model.chunk+13],states,cut)
        rows.append(lp)
        if detach_every and (i+1)%detach_every==0: states = [s.detach() for s in states]
    return torch.cat(rows,dim=1)[:,:total]
