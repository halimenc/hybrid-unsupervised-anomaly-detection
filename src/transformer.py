"""Independently implemented self-supervised masked-feature Transformer."""
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class MaskedTabularTransformer(nn.Module):
    def __init__(self, numeric_count, categorical_sizes=(), width=32, heads=4, layers=2):
        super().__init__()
        self.numeric_count=numeric_count
        self.num_projection=nn.Linear(1,width)
        self.col_embed=nn.Embedding(numeric_count+len(categorical_sizes),width)
        self.cat_embeds=nn.ModuleList(nn.Embedding(size,width) for size in categorical_sizes)
        self.mask_token=nn.Parameter(torch.zeros(width))
        self.missing_token=nn.Parameter(torch.randn(width)*0.02)
        enc=nn.TransformerEncoderLayer(width,heads,dim_feedforward=width*3,dropout=.1,batch_first=True,activation="gelu")
        self.encoder=nn.TransformerEncoder(enc,layers,enable_nested_tensor=False)
        self.reconstruct=nn.Linear(width,1)
    def forward(self,x,mask,cats=None,observed=None):
        n=x.shape[0]
        tokens=self.num_projection(x.unsqueeze(-1))
        tokids=torch.arange(self.numeric_count,device=x.device)
        tokens=tokens+self.col_embed(tokids)[None,:,:]
        if observed is not None:
            tokens=torch.where((~observed.bool()).unsqueeze(-1),self.missing_token[None,None,:],tokens)
        tokens=torch.where(mask.unsqueeze(-1),self.mask_token[None,None,:],tokens)
        if self.cat_embeds:
            if cats is None: raise ValueError("categorical input required")
            context=torch.stack([emb(cats[:,i])+self.col_embed(torch.tensor(self.numeric_count+i,device=x.device)) for i,emb in enumerate(self.cat_embeds)],dim=1)
            tokens=torch.cat((tokens,context),dim=1)
        encoded=self.encoder(tokens)
        return self.reconstruct(encoded[:,:self.numeric_count]).squeeze(-1)


def train_masked(model,x,observed,cats=None,epochs=2,batch_size=128,lr=.001,seed=42):
    torch.manual_seed(seed)
    torch.set_num_threads(min(torch.get_num_threads(),4))
    z=torch.as_tensor(x,dtype=torch.float32)
    obs=torch.as_tensor(observed,dtype=torch.float32)
    c=torch.as_tensor(cats,dtype=torch.long) if cats is not None else torch.zeros((len(z),0),dtype=torch.long)
    loader=DataLoader(TensorDataset(z,obs,c),batch_size=batch_size,shuffle=True)
    optim=torch.optim.AdamW(model.parameters(),lr=lr)
    history=[]
    for ep in range(epochs):
        model.train(); total=0.; weight=0
        for bx,bo,bc in loader:
            present=bo.bool()
            mask=(torch.rand_like(bo)<.22)&present
            # Ensure at least one masked feature for rows with any observations.
            any_present=present.any(1)
            empty=any_present & ~mask.any(1)
            if empty.any():
                indices=empty.nonzero(as_tuple=True)[0]
                for i in indices.tolist():
                    choices=present[i].nonzero(as_tuple=True)[0]
                    mask[i,choices[torch.randint(len(choices),(1,))]]=True
            if not mask.any():continue
            pred=model(bx,mask,bc if bc.shape[1] else None,observed=present)
            loss=((pred-bx)[mask]**2).mean()
            if not torch.isfinite(loss): raise RuntimeError("Nonfinite training loss")
            optim.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),3.);optim.step()
            count=int(mask.sum());total+=float(loss.item())*count;weight+=count
        history.append(total/max(weight,1))
        print(f"    epoch {ep+1}/{epochs}: masked MSE={history[-1]:.5f}",flush=True)
    return history


@torch.no_grad()
def leave_one_out_error(model,x,observed,cats=None,batch_size=256):
    """Score a feature only when it is observed and hidden from the encoder."""
    model.eval()
    all_scores=np.full(x.shape,np.nan,dtype=np.float32)
    for start in range(0,len(x),batch_size):
        stop=start+batch_size
        bx=torch.from_numpy(x[start:stop]).float()
        bo=torch.from_numpy(observed[start:stop]).bool()
        bc=torch.from_numpy(cats[start:stop]).long() if cats is not None else None
        for j in range(x.shape[1]):
            valid=bo[:,j]
            if not valid.any():continue
            mask=torch.zeros_like(bo);mask[:,j]=valid
            predicted=model(bx,mask,bc,observed=bo)
            errors=(predicted[:,j]-bx[:,j])**2
            all_scores[start:stop,j]=torch.where(valid,errors,torch.nan).cpu().numpy()
    return all_scores
