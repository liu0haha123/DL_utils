import nibabel as nib
import numpy as np
import dipy
from dipy.io.streamline import load_trk
AF_LEFT_PEAKS = nib.load("E:\\计算完成\\599469\\tracts\\AF_left_DIRECTIONS.nii.gz")
AF_LEFT_mask = nib.load("E:\\计算完成\\599469\\tracts\\AF_left.nii.gz")
print(AF_LEFT_PEAKS)
print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
print(AF_LEFT_mask)