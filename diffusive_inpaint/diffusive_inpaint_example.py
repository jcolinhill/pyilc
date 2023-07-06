import healpy as hp
import numpy as np

import diffusive_inpaint

file_for_mask =  # fill in filename with a mask defining the region you want to inpaint (regions where the mask is zero will be inpainted in the map)
file_for_map =  # fill in filename where the map you want to inpaint is stored 
new_inpainted_map_filename = # fill in filename where you want to save the new inpainted mask
MASK_VAL = -1.e30

if __name__ == "__main__":

    mask = hp.fitsfunc.read_map(file_for_mask)

    map_to_inpaint  = hp.fitsfunc.read_map(file_for_map)
    map_to_inpaint = hp.remove_monopole(map_to_inpaint)

    map_raw = map_to_inpaint.copy()

    map_raw[np.where(mask == 0.)] = MASK_VAL
    inpainted = diffusive_inpaint.diff_inpaint_vectorized(map_raw, MASK_VAL=MASK_VAL)

    hp.fitsfunc.write_map(new_inpainted_map_filename,inpainted)
  
