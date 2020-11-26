"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from --checkpoints_dir and save the results to --results_dir.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for --num_test images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
import util.util as util

from PIL import Image

class CUTTestWrapper():
    def __init__(self, data_path='/home/hadrien/Bureau/PhD Year 1/Research/LUNG_CT/ct_explorer/tmp_ct_to_us', name='ct2us_qt', phase='test', epoch_to_load=265, gpu=True):
    
        self.opt = TestOptions().parse()
        
        # DEFAULT TEST OPTIONS (C/C)
        self.opt.num_threads = 0   # test code only supports num_threads = 1
        self.opt.batch_size = 1    # test code only supports batch_size = 1
        self.opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
        self.opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
        self.opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
        
        # SPECIFIC OPTIONS
        self.opt.CUT_mode = 'CUT'
        self.opt.dataroot = data_path
        self.opt.name = name
        self.opt.phase = phase
        self.opt.epoch = epoch_to_load
        self.opt.num_test = 999
        
        if gpu==False:
            self.opt.gpu_ids= []
        print(self.opt.gpu_ids, type(self.opt.gpu_ids))
        
        self.dataset = None #create_dataset(opt)
        # self.train_dataset = None #create_dataset(util.copyconf(opt, phase="train"))
        self.model = create_model(self.opt)
        self.model.setup(self.opt)
        self.model.parallelize()

    def update_data(self, rand=None):
        self.dataset = create_dataset(self.opt)
        # self.train_dataset = create_dataset(util.copyconf(self.opt, phase="train"))
        # self.model = create_model(opt)
    
    def generate(self):
        print(len(self.dataset))
        for i, data in enumerate(self.dataset):
            if i == 0:
                self.model.data_dependent_initialize(data)
                # self.model.setup(self.opt)               # regular setup: load and print networks; create schedulers
                # self.model.parallelize()
                # if self.opt.eval:
                #     self.model.eval()
            if i >= self.opt.num_test:  # only apply our model to opt.num_test images.
                break
            self.model.set_input(data)  # unpack data from data loader
            self.model.test()           # run inference
            visuals = self.model.get_current_visuals()  # get image results
            # img_path = self.model.get_image_paths()     # get image paths
            img = util.tensor2im(visuals['fake_B'])

        return img
            # MUST REPLACE THAT LINE : save_images(webpage, visuals, img_path, width=opt.display_winsize)
        

def get_test_prediction(name, data_path, phase='test', epoch_to_load=265):
    opt = TestOptions().parse()
    
    # DEFAULT TEST OPTIONS (C/C)
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    
    # SPECIFIC OPTIONS
    opt.CUT_mode = 'CUT'
    opt.dataroot = data_path
    opt.name = name
    opt.phase = phase
    opt.epoch = epoch_to_load
    opt.num_test = 999
    
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    train_dataset = create_dataset(util.copyconf(opt, phase="train"))
    model = create_model(opt)      # create a model given opt.model and other options
    
    for i, data in enumerate(dataset):
        if i == 0:
            model.data_dependent_initialize(data)
            model.setup(opt)               # regular setup: load and print networks; create schedulers
            model.parallelize()
            if opt.eval:
                model.eval()
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()     # get image paths
        if i % 5 == 0:  # save images to an HTML file
            print('processing (%04d)-th image... %s' % (i, img_path))
        # MUST REPLACE THAT LINE : save_images(webpage, visuals, img_path, width=opt.display_winsize)
        
        
        

if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    train_dataset = create_dataset(util.copyconf(opt, phase="train"))
    model = create_model(opt)      # create a model given opt.model and other options
    # create a webpage for viewing the results
    web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))

    for i, data in enumerate(dataset):
        if i == 0:
            model.data_dependent_initialize(data)
            model.setup(opt)               # regular setup: load and print networks; create schedulers
            model.parallelize()
            if opt.eval:
                model.eval()
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()     # get image paths
        if i % 5 == 0:  # save images to an HTML file
            print('processing (%04d)-th image... %s' % (i, img_path))
        save_images(webpage, visuals, img_path, width=opt.display_winsize)
    webpage.save()  # save the HTML


