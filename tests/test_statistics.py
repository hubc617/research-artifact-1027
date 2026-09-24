import numpy as np
from privileged_emg.aggregate import bootstrap, signflip

def test_statistics_use_subject_vector():
 x=np.arange(1,31,dtype=float)
 assert len(x)==30
 assert bootstrap(x,np.random.default_rng(1),100)[0]==x.mean()
 assert 0 <= signflip(x,np.random.default_rng(2),1000) <= 1
