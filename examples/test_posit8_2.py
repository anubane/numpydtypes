import sys

#sys.path.append('/media/sf_shared_folder/OptimDtypes/dist/')


from posit8_2 import posit8_2 as p8
import numpy as np


def main():
    a = np.array([0.33, 0.25, 0.025, 0.0025, 0.00025], dtype=np.float32)
    print(a, '\n...and dtype: {}'.format(a.dtype))

    b = a.astype(p8)
    print(b, '\n...and dtype: {}'.format(b.dtype))
    
    c = b.astype(np.float32)
    print(c, '\n...and dtype: {}'.format(c.dtype))
    
    d = c.astype(p8)
    print(d, '\n...and dtype: {}'.format(d.dtype))
    
    e = d.astype(np.float32)    
    print(e, '\n...and dtype: {}'.format(e.dtype))

if __name__ == '__main__':
    main()
