"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from '--checkpoints_dir' and save the results to '--results_dir'.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for '--num_test' images and save results to an HTML file.

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
from matplotlib import pyplot as plt
from PIL import Image
import requests
import torchvision.transforms as transforms
from io import BytesIO

def load_image(url):
  response = requests.get(url)
  image = Image.open(BytesIO(response.content)).convert("RGBA")
  image = image.convert("RGB")

  transform = transforms.Compose([
    transforms.Pad(padding=((256 - image.width) // 2,
                  (256 - image.height) // 2),
                  fill=(0, 0, 0)),
    transforms.ToTensor(),
  ])

  torch_image = transform(image)
  return torch_image
if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    
    checkpoint_name = opt.name    
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers

    # load the image
    for url in opt.urls.split(","):
        file_name = os.path.basename(url)

        source_image = load_image(url)

        model.real_A = source_image.unsqueeze(0).to("cuda") * 2. - 1.
        minv, maxv = model.real_A.min(), model.real_A.max()
        print("minv", minv, "maxv", maxv)
    
        model.forward()
        generated_image = model.fake_B.detach().cpu()[0].permute(1, 2, 0)
        generated_image = generated_image * 0.5 + 0.5

        minv, maxv = generated_image.min(), generated_image.max()
        print("minv", minv, "maxv", maxv)

        plt.figure(figsize=(8,8))
        plt.imshow(generated_image)
        plt.axis("off")
        plt.margins(0)
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())  # No x ticks
        plt.gca().yaxis.set_major_locator(plt.NullLocator())  # No y ticks
        plt.gca().set_frame_on(False)                         # No border frame
        plt.savefig(f"./{checkpoint_name}_{file_name}")
        plt.close()



