from __future__ import annotations
import json, random
from pathlib import Path
import numpy as np
import torch
DATA_ROOT: Path | None = None
def configure(data_root):
 global DATA_ROOT; DATA_ROOT = Path(data_root).expanduser().resolve()
def require_data_root():
 if DATA_ROOT is None: raise RuntimeError("Data root is not configured; pass --data-root.")
 return DATA_ROOT
def save_json(path, obj):
 path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(obj,indent=2,sort_keys=True)+"\n")
def read_jsonl(path,index):
 with Path(path).open() as f:
  for i,line in enumerate(f):
   if i==index:return json.loads(line)
 raise IndexError(f"{path}: no row {index}")
def set_seed(seed):
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
 if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
def load_unseen_subjects(subjects,paradigms,emg_notch=True):
 from .dataset import load_data_splits
 _,_,test=load_data_splits(base_dir=str(require_data_root()),paradigms=tuple(paradigms),unseen_subjects=tuple(subjects),train_subjects=(),val_ratio=0.0,emg_notch=emg_notch)
 return test
def load_full_subjects(subjects, paradigms=("overt", "silent_1001"), emg_notch=True):
 from .dataset import load_data_splits
 train,_,_=load_data_splits(base_dir=str(require_data_root()),paradigms=tuple(paradigms),unseen_subjects=(),train_subjects=tuple(subjects),val_ratio=0.0,emg_notch=emg_notch)
 return train
def subset_dict(data,indices):
 indices=np.asarray(indices);return {k:v[indices] for k,v in data.items()}
def classification_metrics(y_true,y_pred,probs=None,n_classes=10):
 y_true=np.asarray(y_true,dtype=np.int64);y_pred=np.asarray(y_pred,dtype=np.int64);cm=np.zeros((n_classes,n_classes),dtype=np.int64)
 for truth,prediction in zip(y_true,y_pred):cm[truth,prediction]+=1
 support,predicted,tp=cm.sum(1),cm.sum(0),np.diag(cm).astype(float);recall=np.divide(tp,support,out=np.zeros(n_classes),where=support>0);precision=np.divide(tp,predicted,out=np.zeros(n_classes),where=predicted>0);f1=np.divide(2*precision*recall,precision+recall,out=np.zeros(n_classes),where=(precision+recall)>0);present=support>0
 result={"n":int(len(y_true)),"accuracy":float(100*np.mean(y_true==y_pred)),"balanced_accuracy":float(100*recall[present].mean()) if present.any() else 0.0,"macro_f1":float(100*f1[present].mean()) if present.any() else 0.0,"weighted_f1":float(100*np.sum(f1*support)/max(len(y_true),1)),"n_present_classes":int(present.sum())}
 if probs is not None:
  probs=np.asarray(probs);result["nll"]=float(-np.log(np.clip(probs[np.arange(len(y_true)),y_true],1e-12,1)).mean())
 return result,cm
