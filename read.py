import h5py
import numpy as np

with h5py.File('data.h5','r') as h5:
    ram0 = h5.get('s')
    act = h5.get('a')
    rew = h5.get('r')
    ram1 = h5.get('sp')
    np_data=np.array(ram0)
    print np_data[7]
