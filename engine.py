import numpy as np
import nibabel as nib
import streamlit as st

from CTHandler import *


def clamp(data, imin=0.0, imax=1.0, omin=0.0, omax=1.0):
    data = data.astype(np.float)
    if imin != None:
        data[data<imin] = imin
    if imax != None:
        data[data>imax] = imax
    data = (data - data.min())/(data.max() - data.min())
    return data * (omax-omin) + omin

def draw_line_spe(data, index, color=(1.0,0.0,0.0), axis=0):
    # print(data.ndim, data.shape)
    if data.ndim == 2:
            data = np.stack((data,)*3, axis=-1)
    # print(data.ndim, data.shape)
    
    if index < 0 or index >= data.shape[not axis]:
        return data
    
    for i in range(data.shape[not axis]-1):
        if axis == 0:
            data[index,i,...] = color
        else:
            data[i,index,...] = color
    
    return data

class Engine():
    def __init__(self):
        self.input_folder = "/home/hadrien/Bureau/PhD Year 1/Research/WEB_US_GEN/DATASET"
        
        self.files_list = list()
        self.ref = None
        self.data = None  
        self.selected_file_path = None
        self.img = None
        self.samp = None
        self.img_us = None
        
        self.slider_value = 0
        
        self.roi_boundaries = [0,0,0,0]
        
        self.gan = None
        
        # self.US_GEN = CUTTestWrapper()

    def attach(self, gan):
        self.gan = gan
    
    def load_CT_list(self):
        files = os.listdir(path=self.input_folder)
        self.files_list = list() # empty list
        for file in files:
            ext = os.path.splitext(file)[1]
            if ext in [".gz", ".nii"]:
                self.files_list.append(file)
        self.files_list.sort()
    
    def load_sel_file(self, selected_file):
        
        if self.data != None:
            old_max = self.data.shape[0]
        
        self.selected_file_path = selected_file
        self.ref = nib.load(os.path.join(self.input_folder,self.selected_file_path))
        self.data = self.ref.get_fdata()
        self.data = copy_mitk_coodinates(self.data)

        # print(self.data.shape)
                
        return self.data.shape[0]-1
    
    def get_image_ax_at_index(self, index, imin=100.0, imax=300.0):
        self.img = clamp(self.data[index,:,:], imin, imax, omin=0.0, omax=1.0)
        self.slider_value = index
        # print("img_ax",self.img.shape, self.img.min(), self.img.max())
        # self.img = scipy.ndimage.zoom(self.img, (0.5,0.5), order=1)
        return self.img
    
    def rotate_ax_image(self, angle):
        self.img = rotate_image(self.img, -angle)
        center = self.img.shape[0]/2
        self.img = self.img[int(center-256):int(center+256), int(center-256):int(center+256)]
        return self.img
    
    def get_image_sa_at_index(self, index, imin=100.0, imax=300.0):
        
        self.img_sa = clamp(self.data[:,:,int(self.data.shape[2]/2)], imin=None, imax=None, omin=0.0, omax=1.0)
        self.img_sa = draw_line_spe(self.img_sa, int(index),color=(1.0, 0., 0.))
        # print("img_sa",self.img_sa.shape, self.img_sa.min(), self.img_sa.max())

        i = 2
        j = 0
        affine_scale = np.array([self.ref.affine[i,i], self.ref.affine[j,j]])
        affine_scale = affine_scale/affine_scale.min()
        affine_scale = np.array([affine_scale[0], affine_scale[1], 1])
        self.img_sa = scipy.ndimage.zoom(self.img_sa, affine_scale, order=1)

        # self.img_sa = draw_line_spe(self.img_sa, int(index*affine_scale[0]),color=(1.0, 0., 0.))
                
        # print("img_sa",self.img_sa.shape, self.img_sa.min(), self.img_sa.max())
        
        center = self.img_sa.shape[1]/2
        half_margin = 1/3*self.img_sa.shape[0]
        begin = int(center-half_margin)
        end   = int(center+half_margin)
        
        # print(begin, end)
        
        return self.img_sa[:,begin:end,:]

    def draw_ROI(self, horiz_start, horiz_end, verti_start, verti_end):
        
        self.roi_boundaries = [horiz_start, horiz_end, verti_start, verti_end]
        
        self.sample_squarre_ROI(self.slider_value)
        
        self.img = draw_line_spe(self.img, horiz_start, color=(1.0,0.0,0.0), axis=1)
        self.img = draw_line_spe(self.img, horiz_end,   color=(1.0,0.0,0.0), axis=1)
        self.img = draw_line_spe(self.img, verti_start, color=(1.0,0.0,0.0), axis=0)
        self.img = draw_line_spe(self.img, verti_end,   color=(1.0,0.0,0.0), axis=0)
        
        return self.img
    
    def sample_squarre_ROI(self, index, imin=100.0, imax=300.0):
        w_min = self.roi_boundaries[2]
        w_max = self.roi_boundaries[3]
        h_min = self.roi_boundaries[0]
        h_max = self.roi_boundaries[1]
        
        w_d = w_max - w_min
        h_d = h_max - h_min
        
        padding_a = padding_b = int(abs(w_d-h_d)/2)
        if int(abs(w_d-h_d)/2) != (abs(w_d-h_d)/2):
            padding_b += 1 
            
        overflow_h_min = 0 if (h_min-padding_a) == abs(h_min-padding_a) else abs(h_min-padding_a)
        overflow_h_max = 0 if (h_max+padding_b) == abs(h_max+padding_b) else abs(h_max+padding_b)
        overflow_w_min = 0 if (w_min-padding_a) == abs(w_min-padding_a) else abs(w_min-padding_a)
        overflow_w_max = 0 if (w_max+padding_b) == abs(w_max+padding_b) else abs(w_max+padding_b)
        
        if w_d > h_d:
            h_min = h_min-padding_a + overflow_h_min - overflow_h_max
            h_max = h_max+padding_b + overflow_h_min - overflow_h_max
        else:
            w_min = w_min-padding_a + overflow_w_min - overflow_w_max
            w_max = w_max+padding_b + overflow_w_min - overflow_w_max
            
        
        
        # print("d",w_d, h_d, "p", padding_a, padding_b, "w", w_min,w_max, "h",h_min, h_max)
        
        self.samp = self.img[w_min:w_max,h_min:h_max]
        print(self.samp.shape)
        return self.samp
    
    # @st.cache(suppress_st_warning=True)
    def gen_us(self):
        
        # save image
        # print(self.samp.shape)
        path = "tmp_ct_to_us/testA/ct.png"
        zoom = 256/self.samp.shape[0]
        to_save = (scipy.ndimage.zoom(self.samp, (zoom,zoom), order=1) * 255).astype(np.uint8)
        # print(to_save.shape)
        Image.fromarray(to_save).save(path)
        
        # print("in gen us")
        # path = "tmp_ct_to_us/testA/ct.png"
        # zoom = 256/img.shape[0]
        # to_save = (scipy.ndimage.zoom(img, (zoom,zoom), order=1) * 255).astype(np.uint8)
        # print(to_save.shape)
        # Image.fromarray(to_save).save(path)
        
        # call gan
        self.gan.update_data()
        self.img_us = self.gan.generate()

        # print(rand)
        
        return self.img_us
        