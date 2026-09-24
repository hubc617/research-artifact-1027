"""Secondary analyses for the validated experiment pipeline."""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from . import common, chain
from .dataset import BCIDataset
from .model_stage1 import RobustEMGPretrainer
from .model_stage2 import EEGMAEAlignerPretrainer
from .model_stage3 import EEGDownstreamClassifier, EMGDownstreamClassifier, BimodalEEGEMGClassifier
from .label_simplex import EEGIndexedDataset, load_partition_indices, stage3_data_paths

EEG_CHANNELS = ['FP1','FPZ','FP2','AF3','AF4','F7','F5','F3','F1','FZ','F2','F4','F6','F8','FT7','FC5','FC3','FC1','FCZ','FC2','FC4','FC6','FT8','T7','C5','C3','C1','CZ','C2','C4','C6','T8','TP7','CP5','CP3','CP1','CPZ','CP2','CP4','CP6','TP8','P7','P5','P3','P1','PZ','P2','P4','P6','P8','PO7','PO5','PO3','POZ','PO4','PO6','PO8','O1','OZ','O2']
OUTER_RING_WIDE = ['FP1','FPZ','FP2','AF3','AF4','F7','F5','F6','F8','FT7','FC5','FC6','FT8','T7','C5','C6','T8','TP7','CP5','CP6','TP8','P7','P5','P6','P8','PO7','PO5','PO6','PO8','O1','OZ','O2']

def load_state(model, path):
 state=torch.load(path,map_location='cpu'); model.load_state_dict({k.replace('module.','',1):v for k,v in state.items()},strict=True)
def device():
 if not torch.cuda.is_available(): raise RuntimeError('CUDA is required for training secondary analyses')
 return torch.device('cuda')
def new_output(path):
 path=Path(path)
 if path.exists(): raise FileExistsError(f'refusing to overwrite existing output: {path}')
 path.parent.mkdir(parents=True,exist_ok=True)
 return path
def loader(data,batch,shuffle=False,seed=0):
 dataset=data if isinstance(data,Dataset) else BCIDataset(data)
 return DataLoader(dataset,batch_size=batch,shuffle=shuffle,generator=torch.Generator().manual_seed(seed) if shuffle else None,num_workers=0,drop_last=shuffle)
def eeg_scale(drop, dev):
 scale=torch.ones(1,60,1,device=dev)
 for channel in drop: scale[0,EEG_CHANNELS.index(channel),0]=0
 return scale
@torch.no_grad()
def evaluate(model, data, modality, dev, scale=None):
 ys=[]; ps=[]; qs=[]; model.eval()
 for batch in loader(data,64):
  if len(batch)==2: eeg,y=batch; emg=None
  else: eeg,emg,_,y,_=batch
  eeg=eeg.to(dev); emg=emg.to(dev) if emg is not None else None
  if scale is not None: eeg=eeg*scale
  if modality=='eeg': out=model(eeg)
  elif modality=='emg': out=model(emg)
  else: out=model(eeg,emg)
  p=torch.softmax(out['logits'],1);ys.append(y.numpy());ps.append(p.argmax(1).cpu().numpy());qs.append(p.cpu().numpy())
 return common.classification_metrics(np.concatenate(ys),np.concatenate(ps),np.concatenate(qs))[0]
@torch.no_grad()
def evaluate_pretrainer(model, data, stage, dev):
 ys=[];ps=[];qs=[];model.eval()
 if stage==1: model.emg_mask_ratio=0.
 else: model.eeg_mask_ratio=0.
 for eeg,emg,a,y,o in loader(data,64):
  if stage==1: out=model(emg.to(dev),a.to(dev),y.to(dev),o.to(dev))
  else: out=model(eeg.to(dev),emg.to(dev),y.to(dev),use_distill=False)
  p=torch.softmax(out['logits'],1);ys.append(y.numpy());ps.append(p.argmax(1).cpu().numpy());qs.append(p.cpu().numpy())
 return common.classification_metrics(np.concatenate(ys),np.concatenate(ps),np.concatenate(qs))[0]
def train_downstream(train,val,final_test_factory,stage1,stage2,modality,epochs,seed,drop=(),encoder_lr=3e-6,classifier_lr=1e-4):
 dev=device(); common.set_seed(seed)
 if modality=='eeg': model=EEGDownstreamClassifier(stage2,10).to(dev)
 elif modality=='emg': model=EMGDownstreamClassifier(stage1,10).to(dev)
 else: model=BimodalEEGEMGClassifier(stage2,stage1,10).to(dev)
 if modality=='eeg': encoder_params=model.eeg_encoder.parameters()
 elif modality=='emg': encoder_params=model.emg_encoder.parameters()
 else: encoder_params=list(model.eeg_encoder.parameters())+list(model.emg_encoder.parameters())
 opt=optim.AdamW([{'params':encoder_params,'lr':encoder_lr},{'params':model.classifier.parameters(),'lr':classifier_lr}],weight_decay=.05);sch=optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs,eta_min=1e-6); scale=eeg_scale(drop,dev) if drop else None
 train_loader=loader(train,8,True,seed); best=-1.;best_state=None
 for _ in range(epochs):
  model.train()
  for batch in train_loader:
   if len(batch)==2: eeg,y=batch; emg=None
   else: eeg,emg,_,y,_=batch
   eeg,y=eeg.to(dev),y.to(dev); emg=emg.to(dev) if emg is not None else None
   if scale is not None:eeg=eeg*scale
   out=model(eeg,y,mask_ratio=0.) if modality=='eeg' else (model(emg,y,mask_ratio=0.) if modality=='emg' else model(eeg,emg,y))
   opt.zero_grad(set_to_none=True);out['loss'].backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
  sch.step();score=evaluate(model,val,modality,dev,scale)['balanced_accuracy']
  if score>best: best=score;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
 if best_state is None: raise RuntimeError('no validation checkpoint selected')
 model.load_state_dict(best_state)
 test=final_test_factory()
 return best,evaluate(model,test,modality,dev,scale),model

def eeg_train_validation_and_test_factory(subject,paradigm,fold):
 eeg_path,label_path=stage3_data_paths(common.require_data_root(),subject,paradigm)
 train_idx=load_partition_indices(SPLIT_ROOT,subject,paradigm,fold,'train')
 validation_idx=load_partition_indices(SPLIT_ROOT,subject,paradigm,fold,'validation')
 if np.intersect1d(train_idx,validation_idx).size: raise AssertionError('train/validation overlap')
 train=EEGIndexedDataset(eeg_path,label_path,train_idx,'train')
 validation=EEGIndexedDataset(eeg_path,label_path,validation_idx,'validation')
 def final_test_factory():
  test_idx=load_partition_indices(SPLIT_ROOT,subject,paradigm,fold,'final_test')
  if np.intersect1d(train_idx,test_idx).size or np.intersect1d(validation_idx,test_idx).size: raise AssertionError('final-test overlap')
  return EEGIndexedDataset(eeg_path,label_path,test_idx,'final_test')
 return train,validation,final_test_factory
def subject_split(subject,paradigm,fold): return chain.load_subject_split(subject,paradigm,fold)
def zero_shot(args):
 new_output(args.output)
 dev=device(); data=common.load_unseen_subjects([args.subject],[args.paradigm]); base={'subject':args.subject,'paradigm':args.paradigm,'stage1_ckpt':args.stage1_checkpoint,'stage2_ckpt':args.stage2_checkpoint}
 s1=RobustEMGPretrainer(256,10,.75).to(dev);load_state(s1,args.stage1_checkpoint)
 s2=EEGMAEAlignerPretrainer(args.stage1_checkpoint,256,10,.75,.3).to(dev);load_state(s2,args.stage2_checkpoint)
 result={**base,'stage1':evaluate_pretrainer(s1,data,1,dev),'stage2':evaluate_pretrainer(s2,data,2,dev)};common.save_json(args.output,result)
def adaptation(args):
 new_output(args.output)
 data=common.load_unseen_subjects([args.subject],[args.paradigm]); labels=data['labels']; seed=args.seed+1000*args.inner_fold
 pool,test_idx=list(StratifiedKFold(5,shuffle=True,random_state=20260801).split(np.zeros(len(labels)),labels))[args.inner_fold]
 train_local,val_local=next(StratifiedShuffleSplit(1,test_size=.15,random_state=seed).split(np.zeros(len(pool)),labels[pool])); train,val=(common.subset_dict(data,x) for x in (pool[train_local],pool[val_local]))
 best,metrics,_=train_downstream(train,val,lambda:common.subset_dict(data,test_idx),args.stage1_checkpoint,args.stage2_checkpoint,'eeg',args.epochs,seed,encoder_lr=1e-6,classifier_lr=1e-5)
 common.save_json(args.output,{'subject':args.subject,'paradigm':args.paradigm,'inner_fold':args.inner_fold,'seed':args.seed,'n_train':len(train['labels']),'n_val':len(val['labels']),'n_test':len(test['labels']),'best_val_balanced_accuracy':best,**{f'test_{k}':v for k,v in metrics.items()}})
def channel(args):
 new_output(args.output)
 train,val,final_test_factory=eeg_train_validation_and_test_factory(args.subject,args.paradigm,args.fold);drop=OUTER_RING_WIDE if args.preset=='28-channel' else ()
 best,metrics,_=train_downstream(train,val,final_test_factory,None,args.stage2_checkpoint,'eeg',args.epochs,args.seed,drop)
 common.save_json(args.output,{'subject':args.subject,'paradigm':args.paradigm,'fold':args.fold,'preset':args.preset,'n_channels':60-len(drop),'dropped_channels':drop,'mask_applied_roles':['train','validation','final_test'],'stage2_unchanged':True,'stage3_modality':'EEG_only','final_test_loader_constructions':1,'final_test_loader_traversals':1,'test_used_for_selection':False,'best_val_balanced_accuracy':best,**{f'test_{k}':v for k,v in metrics.items()}})
def modality(args):
 new_output(args.output)
 train,val,test=subject_split(args.subject,args.paradigm,args.fold);best,metrics,_=train_downstream(train,val,lambda:test,args.stage1_checkpoint,args.stage2_checkpoint,args.modality,args.epochs,args.seed)
 common.save_json(args.output,{'subject':args.subject,'paradigm':args.paradigm,'fold':args.fold,'modality':args.modality,'best_val_balanced_accuracy':best,**{f'test_{k}':v for k,v in metrics.items()}})
def smoothgrad(args):
 new_output(args.output)
 _,_,test=subject_split(args.subject,args.paradigm,args.fold);dev=device();model=EEGDownstreamClassifier(args.stage2_checkpoint,10).to(dev);load_state(model,args.stage3_checkpoint);model.eval();sums=torch.zeros(60,device=dev);n=0
 for eeg,_,_,labels,_ in loader(test,32):
  eeg,labels=eeg.to(dev),labels.to(dev);trial=torch.zeros_like(eeg);sd=eeg.std(-1,keepdim=True).clamp_min(1e-6)
  for _ in range(args.samples):
   x=(eeg+args.noise_ratio*sd*torch.randn_like(eeg)).detach().requires_grad_(True);score=model(x)['logits'].gather(1,labels[:,None]).sum();trial+=torch.autograd.grad(score,x)[0].abs()
  trial=(trial/args.samples).mean(-1);trial=trial/trial.sum(1,keepdim=True).clamp_min(1e-12);sums+=trial.sum(0);n+=len(eeg)
 rows=[{'channel':ch,'relative_attribution':float(sums[i].cpu()/n)} for i,ch in enumerate(EEG_CHANNELS)]
 with Path(args.output).open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
def fig2(args):
 new_output(args.output)
 import matplotlib.pyplot as plt
 import mne
 frame=[]
 for path in args.inputs:
  df=np.genfromtxt(path,delimiter=',',names=True,dtype=None,encoding='utf8');frame.append({r['channel']:float(r['relative_attribution']) for r in df})
 values=np.mean([[item[ch] for ch in EEG_CHANNELS] for item in frame],axis=0);info=mne.create_info(EEG_CHANNELS,256,'eeg');info.set_montage(mne.channels.make_standard_montage('standard_1020'),match_case=False,on_missing='warn');fig,ax=plt.subplots(figsize=(5,4));mne.viz.plot_topomap(values,info,axes=ax,show=False);fig.savefig(args.output,dpi=220,bbox_inches='tight');plt.close(fig)
def parser():
 p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='kind',required=True)
 def shared(x,stage1=True):
  x.add_argument('--data-root',required=True);x.add_argument('--split-root',required=True);x.add_argument('--stage2-checkpoint',required=True);x.add_argument('--stage1-checkpoint',required=stage1);x.add_argument('--subject',required=True);x.add_argument('--paradigm',choices=('overt','silent_1001'),required=True);x.add_argument('--output',required=True)
 z=s.add_parser('zero-shot');shared(z);z.set_defaults(func=zero_shot)
 a=s.add_parser('adaptation');shared(a);a.add_argument('--inner-fold',type=int,required=True);a.add_argument('--seed',type=int,default=42);a.add_argument('--epochs',type=int,default=100);a.set_defaults(func=adaptation)
 c=s.add_parser('channel-removal');shared(c,False);c.add_argument('--fold',type=int,required=True);c.add_argument('--seed',type=int,default=42);c.add_argument('--epochs',type=int,default=100);c.add_argument('--preset',choices=('all','28-channel'),required=True);c.set_defaults(func=channel)
 m=s.add_parser('modality');shared(m);m.add_argument('--fold',type=int,required=True);m.add_argument('--seed',type=int,default=42);m.add_argument('--epochs',type=int,default=100);m.add_argument('--modality',choices=('emg','bimodal'),required=True);m.set_defaults(func=modality)
 g=s.add_parser('smoothgrad');shared(g,False);g.add_argument('--stage3-checkpoint',required=True);g.add_argument('--fold',type=int,required=True);g.add_argument('--samples',type=int,default=16);g.add_argument('--noise-ratio',type=float,default=.1);g.set_defaults(func=smoothgrad)
 f=s.add_parser('fig2');f.add_argument('--inputs',nargs='+',required=True);f.add_argument('--output',required=True);f.set_defaults(func=fig2)
 return p
def main(argv=None):
 args=parser().parse_args(argv);common.configure(args.data_root) if hasattr(args,'data_root') else None;chain.configure(args.split_root) if hasattr(args,'split_root') else None;args.func(args)
