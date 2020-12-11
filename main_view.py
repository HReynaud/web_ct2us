import streamlit as st
import numpy as np
import pandas as pd
import time

from engine import Engine
from test import CUTTestWrapper

def get_model():
    return CUTTestWrapper(data_path='tmp_ct_to_us', gpu=False)

"""
# Cardiac CT Scan to Ultrasound
Welcome to this demo. Here we present a pipeline to:
- 1) manually sample the heart in a full-body CT scan
- 2) rotate and tilt that sample to get an orientation similar to a 4-chamber view ultrasound
- 3) apply style transfer (using a [GAN](https://github.com/taesungp/contrastive-unpaired-translation)) on the sampled CT image to simulate a 4-chamber view ultrasound


## How it works:
*Note:* The default parameters will give an example of a well calibrated sample. Simply click on the "Generate US from sample" button in the left panel to try it.
- Step 1: selection of the full body CT scan. Pick one from the list. 
- Step 2: selection of the axial slice index (0: head, max: foot). The position of the slice can be visualised on the sagital view on the right.
- Step 3: 
    - (1) rotates the image to match the position of the heart in an ultrasound (Ventricles top, Atriums bottom)
    - (2) tilts the acquisition plan on the saggital axis (computed before rotation)
    - (3) tilts the acquisition plan on the coronal axis (computed before rotation)
    - *Note:* The orientation is shown in the sagittal view. The saggital tilt is shown by the orientation of the red lines and the coronal tilt by the space interval between the light and dark red lines (light = left, dark = right) 
- Step 4: select the region of the heart to imitate the relative size of the heart in an ultrasound
"""

gan = get_model()

eng = Engine()
eng.load_CT_list()
eng.attach(gan)

selected_ct = st.sidebar.selectbox( 'Step 1: Select a CT Scan:', eng.files_list )
slide_max = eng.load_sel_file(selected_ct)

ax_index = st.sidebar.slider( 'STEP 2: Select axial slice (top to bottom)', 0, slide_max, 70, 1)
st.sidebar.text('STEP 3: Orient the slice')
rotation = st.sidebar.slider( '1: Rotate Axial Slice', -180, 180, 44, 1)
tilt1 = st.sidebar.slider( '2: Saggital tilt', -45.0, 45.0, 0.0, 0.1)
tilt2 = st.sidebar.slider( '3: Coronal tilt', -45.0, 45.0, 0.0, 0.1)
st.sidebar.text('STEP 4: Sample heart')
horizontal_cut = st.sidebar.slider( ' Select horizontal sampling:', 0, 511, (160,420), 1)
vertical_cut   = st.sidebar.slider( 'Select vertical sampling:',   0, 511, (60,320), 1)

# st.sidebar.text('WIP:')
# value_range = st.sidebar.slider( 'Select a range of values (do not use)', eng.data.min(), eng.data.max(), (np.float(eng.data.min()), np.float(eng.data.max())))

# img_ax = eng.get_image_ax_at_index(ax_index, imin=eng.data.min(), imax=eng.data.max())
img_ax = eng.get_tilted_img_at_index(tilt1, tilt2, ax_index, imin=eng.data.min(), imax=eng.data.max())
img_ax = eng.rotate_ax_image(rotation)
img_ax = eng.draw_ROI(horizontal_cut[0], horizontal_cut[1], vertical_cut[0], vertical_cut[1])

img_sa = eng.get_image_sa_at_index(ax_index, imin=eng.data.min(), imax=eng.data.max(), y=tilt1, z=tilt2)


img_samp = eng.samp
img_us = np.zeros((512,512))

if st.sidebar.button("Generate US from sample"):
    img_us = eng.gen_us()

col1, col2 = st.beta_columns([3,2])
with col1:
    st.header("Axial slice "+str(ax_index))
    st.image(img_ax, use_column_width=True)
    
    st.header("Sampled slice")
    st.image(img_samp, use_column_width=True)
    
    st.header("Style-Transfered to US")
    st.image(img_us, use_column_width=True)
    
with col2:
    st.header("Saggital view")
    st.image(img_sa, use_column_width=True)
