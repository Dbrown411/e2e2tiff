import numpy as np
import h5py, tempfile
import os
from tqdm import tqdm
from skimage.registration import phase_cross_correlation
from cv2 import warpAffine, INTER_LANCZOS4

def make_M(xyshift):
    xshift,yshift = xyshift
    return np.float32([[1,0,yshift*0],[0,1,xshift]])

def create_disk_buffer(vol,name='volume'):
    _,x,y = vol.shape
    chunksize = (1,x,y)
    tf = h5py.File(tempfile.TemporaryFile(), "w")
    buffer = tf.create_dataset(name,
                                shape=vol.shape, 
                                dtype = np.uint16,
                                chunks=chunksize)
    return buffer

def shift_frame(frame,shifts):
    rows,cols = np.shape(frame)
    M = make_M(shifts)
    shifted = warpAffine(frame,M,(cols,rows),flags=INTER_LANCZOS4,borderMode=0)
    return shifted   
    
def register_volume(vol):
    upsampling = 100
    noShift = (0,0)
    scan_count,x,y = vol.shape
    register_order = list(range(1,scan_count))
    register_order.append(0)
    reg_vol = create_disk_buffer(vol,'reg_vol')
    frameshifts = np.empty(scan_count, dtype=object)
    for i,s in enumerate(tqdm(register_order,leave=False, desc='Registering OCT')):
        if i==0:
            shifted = vol[s,:,:]
            shift = noShift
        else:
            fixed_index = s-1
            if i==scan_count-1:
                fixed_index = s+1
            prevFixed =reg_vol[fixed_index,:,:]
            moving = vol[s,:,:]
            shift, _, _ = phase_cross_correlation(prevFixed, moving, upsample_factor=upsampling,overlap_ratio=0.7)
            shifted = shift_frame(moving,shift)
        reg_vol[s,:,:] = shifted
        frameshifts[s] = tuple(shift)
    return reg_vol, frameshifts