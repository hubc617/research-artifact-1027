import numpy as np
from privileged_emg.chain import _derange_emg_within_class

def test_within_class_derangement_has_no_fixed_points():
 data={"labels":np.array([0,0,0,1,1,1]),"emg":np.arange(6)[:,None]}
 shuffled,audit=_derange_emg_within_class(data,np.random.default_rng(7),"S01","overt")
 assert audit["fixed_points"]==0 and audit["label_mismatches"]==0
 assert sorted(shuffled["emg"].ravel())==list(range(6))
