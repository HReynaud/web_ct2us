import numpy as np
import scipy.ndimage
import nibabel as nib
import os, shutil
from PIL import Image
from datetime import datetime
import sys


AXIS_NAMES = ["axial", "coronal", "sagital"]
image_format = '.jpg'


def timestamp():
    return str("["+str(datetime.now().strftime("%H:%M:%S"))+"]")

def to_str(val, num_char):
    vs = str(val)
    while len(vs) < num_char:
        vs = '0'+vs
    return vs

def get_axis_index(axis, numerical=True):
    if type(axis) == str:
        index = AXIS_NAMES.index(axis.lower())
    elif type(axis) == int:
        if axis in [0,1,2]:
            index = axis
            axis  = AXIS_NAMES[index]
        else:
            raise ValueError(f'{axis} is not a valid axis value.')
    else:
        raise ValueError(f'{axis} is not a valid axis value.')
    
    if numerical:
        return index
    else:
        return axis
    
def save_image(data, dest):    
    Image.fromarray(data.astype(np.uint8)).save(dest)
    
def normalise_image(data, vmin=0, vmax=100):
    data = ((data - vmin)/(vmax - vmin))*255
    data = data*(data >= 0)*(data <= 255)
    return data

def rotate_image(data, angle, axes=(0,1)):
    return scipy.ndimage.rotate(data, angle, axes=axes, reshape=True, output=None, order=1, mode='constant', cval=0.0, prefilter=True)   

def draw_line(data, axis, index, color):
    if data.ndim == 2 and len(color) == 3:
        data = np.stack((data,)*3, axis=-1)
    
    if index < 0 or index >= data.shape[not axis]:
        return data
    
    for i in range(data.shape[axis]-1):
        if axis == 0:
            data[index,i,...] = color
        else:
            data[i,index,...] = color
    
    return data
        
def extract_all_slices(data, axis, in_file, dest_folder, step=1):    
    index = get_axis_index(axis)
    axis  = get_axis_index(axis, False)
    
    axis_folder = os.path.join(dest_folder)
    if not os.path.exists(axis_folder):
        os.mkdir(axis_folder)
    
    identifier = os.path.split(in_file)[1][10:14]
    
    for i in range(0,data.shape[index],step):
        dest = os.path.join(axis_folder, identifier+"-"+to_str(i,3)+image_format)
        if index == 0:
            save_image(data[i,:,:], dest)
        elif index == 1:
            save_image(data[:,i,:], dest)
        elif index == 2:
            save_image(data[:,:,i], dest)

def copy_mitk_coodinates(data):
    data = np.rot90(data, k=1, axes=(1,2))
    data = np.rot90(data, k=3, axes=(0,1))
    data = np.rot90(data, k=1, axes=(1,2))
    data = np.flip(data, axis=2)
    return data

def affine_scaling(data, coefs):
    coefs = np.array(coefs)
    affine_scale = coefs/coefs.min() # min value becomes 1 => no data loss
    return scipy.ndimage.zoom(data, affine_scale, order=1)

def deform_to_US(data, a, b, c):
    # Must be build interatively, rotation breaks axis order
    data = rotate_image(data, a, axes=(0,1))
    data = rotate_image(data, b, axes=(0,2)) 
    data = rotate_image(data, c, axes=(1,2))
    return data

def us_mask(data, sx, ex, sy, ey, sz, ez):
    return data[int(sx):int(ex), int(sy):int(ey), int(sz):int(ez)]

def clear_folder(path):
    folder = path
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

def timelog(verbose, message):
    if verbose:
        print(timestamp(), message)

def apply_on_whole_folder(in_path, out_path, verbose=0):
    files = os.listdir(in_path)
    i = 0
    for f in files:
        print(i, f)
        i = i+1
        if os.path.splitext(f)[1] == '.gz':
            process_data(os.path.join(in_path,f), out_path)

def process_data(in_path, out_path, verbose=0):
    
    ref = nib.load(in_path)
    data = ref.get_fdata()  
    timelog(verbose, data.shape)

    vmin  = -150
    vmax  =  150
    
    timelog(verbose, "rotating...")
    data = copy_mitk_coodinates(data)
    
    timelog(verbose,  "masking...")
    data = us_mask(data, 0.35*data.shape[0], 0.5*data.shape[0], 0.3*data.shape[1], 0.65*data.shape[1], 0.35*data.shape[2], 0.7*data.shape[2])
    
    timelog(verbose,  "normalising...")  
    data = normalise_image(data, vmin, vmax)
    
    timelog(verbose,  "rotating...")
    data = rotate_image(data, 45, axes=(1,2))
    
    timelog(verbose, "padding...")
    data = np.pad(data, ((0,0), (1,2), (1,2)))
    
    timelog(verbose, "sampling...")
    extract_all_slices(data, 0, in_path, out_path)

if __name__ == '__main__':    
    # Parameters
    verbose = 0
    # in_path = "/home/hadrien/Bureau/PhD Year 1/Research/LUNG_CT/LIDC-IDRI-000"+str(lung_id)+".nii.gz"
    # out_path = "/home/hadrien/Bureau/PhD Year 1/Research/LUNG_CT/slices"

    in_path = sys.argv[1]
    out_path = sys.argv[2]
    
    apply_on_whole_folder(in_path, out_path)


    # ref = nib.load(in_path)
    # data = ref.get_fdata()  
    # timelog(verbose, data.shape)

    # vmin  = -150
    # vmax  =  150
    # angle =  40
    # affine_scale = [ref.affine[i,i] for i in range(3)]
    
    # # clear_folder(slice_folder)
    
    # # FUNCTION CALL IN THIS ORDER ! (affine scale before mitk coordinates)

    # # timelog(verbose, "scaling...")
    # # data = affine_scaling(data, affine_scale)
    # # print("After scaling",data.shape)
    
    # timelog(verbose, "rotating...")
    # data = copy_mitk_coodinates(data)
    
    # # timelog(verbose, "deforming...")
    # # data = deform_to_US(data, 0,0,0)
    
    # timelog(verbose,  "masking...")
    # data = us_mask(data, 0.35*data.shape[0], 0.5*data.shape[0], 0.3*data.shape[1], 0.65*data.shape[1], 0.35*data.shape[2], 0.7*data.shape[2])
    
    # timelog(verbose,  "normalising...")  
    # data = normalise_image(data, vmin, vmax)
    
    # timelog(verbose,  "rotating...")
    # data = rotate_image(data, 45, axes=(1,2))
    
    # timelog(verbose, "sampling...")
    # extract_all_slices(data, 0, out_path)