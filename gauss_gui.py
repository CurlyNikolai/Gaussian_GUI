
import numpy as np
import torch
import torchvision.transforms as transforms
import threading
import imgui
import math
import matplotlib.pyplot as plt
import glfw
from PIL import Image
from scipy.signal import convolve2d
from os import listdir
from gaussian_filter import * 
from viewer import viewer
to_tensor = transforms.ToTensor()
# This function will be called by the viewer for every frame.
# v is a reference to the viewer itself (the footprint has to be this)
def loop(v):

	imgui.begin('Original image')
	v.draw_image('original_image')
	
	v.new_img, v.file_dropdown_current = imgui.combo(
		"", v.file_dropdown_current, v.files
	)

	if v.new_img:
		print(v.files[v.file_dropdown_current])
		img = Image.open("images/"+v.files[v.file_dropdown_current])
		img = np.asarray(img)

		v.img = to_tensor(img.copy())
		v.img = v.img.permute(1, 2, 0)
		v.img = v.img.to(torch.device(device))
		v.img_conv = v.img.clone()

		v.px = img.shape[0]//2
		v.py = img.shape[1]//2

	imgui.end()

	imgui.begin('Kernel')
	v.angle = imgui.slider_float('Angle', v.angle, 0.0, 360.0)[1]
	v.sx = imgui.slider_float('sigma_u', v.sx, 1.0, 5.0)[1]
	v.sy = imgui.slider_float('sigma_v', v.sy, 1.0, 5.0)[1]
	v.draw_image('kernel', scale=20)
	imgui.end()

	imgui.begin('Filtered image')
	
	# Filtered image child
	imgui.begin_child('child')
	v.draw_image('filtered_image')
	
	# Check if new pixel is clicked on filtered image
	if (imgui.core.is_item_clicked()):

		# Track mouse position on filtered image
		new_px = imgui.core.get_mouse_position().x - imgui.core.get_window_position().x
		new_py = imgui.core.get_mouse_position().y - imgui.core.get_window_position().y
		new_px = int(max(0.0, min(new_px, v.img.shape[0]-1)))
		new_py = int(max(0.0, min(new_py, v.img.shape[1]-1)))

		# Update context pixel
		v.px = new_px
		v.py = new_py

	imgui.end_child()
	
	imgui.end()

	imgui.begin('Selected pixel surrounding')
	v.draw_image('pixel_image', scale=10)
	imgui.end()

	imgui.begin('Selected pixel variations')
	imgui.text('Pixel colourmap')
	imgui.text('Axes: (sigma_u, sigma_v)')
	v.draw_image('draw_pixel_variations', scale=20)
	imgui.end()


def draw_original(v, device):
	while not v.quit:
		v.upload_image('original_image', v.img)
			

def draw_kernel(v, device):
	while not v.quit:

		v.filter = GaussianFilter(3, v.size, v.sx, v.sy, (v.angle / 360.0) * 2 * math.pi)
		v.kernel = v.filter.weight

		img = v.kernel.permute(2, 3, 1, 0)[:, :, 0, 0]
		img = img.to(torch.device(device))

		# The factor 10 is just for scaling the pixel colours to be better visible
		v.upload_image('kernel', 10 * img)


def draw_filtered(v, device):
	while not v.quit:
		img_torch = v.img.permute(2,0,1)
		img_torch.unsqueeze_(0)
		img_torch = F.pad(img_torch, (v.size//2, v.size//2, v.size//2, v.size//2))
		v.img_conv = v.filter(img_torch).permute(2, 3, 0, 1)[:, :, 0]
		v.upload_image('filtered_image', v.img_conv)

def draw_pixel_surrounding(v, device):
	while not v.quit:
		v.upload_image('pixel_image', v.img_conv[v.py-v.size//2:v.py+v.size//2+1, v.px-v.size//2:v.px+v.size//2+1])

def draw_multiple_pixel_convolutions(v, device):
	sigma_space = np.arange(1.0, 5.0, 0.5)
	while not v.quit:
		img = v.img[v.py-v.size//2:v.py+v.size//2+1, v.px-v.size//2:v.px+v.size//2+1].permute(2, 0, 1)
		img.unsqueeze_(0)
		img = F.pad(img, (v.size//2, v.size//2, v.size//2, v.size//2))
		img_show = np.zeros((sigma_space.shape[0], sigma_space.shape[0], 3))
		
		for count_x, s_x in enumerate(sigma_space):
			for count_y, s_y in enumerate(sigma_space):
				filter = GaussianFilter(3, v.size, s_x, s_y, (v.angle / 360.0) * 2 * math.pi)
				img_conv = filter(img).permute(2, 3, 0, 1)[:, :, 0]
				img_show[img_show.shape[1]-1-count_y, count_x] = img_conv[img_conv.shape[0]//2, img_conv.shape[1]//2].cpu()
		
		img_torch = to_tensor(img_show.copy()).permute(1, 2, 0)
		img_torch = img_torch.to(torch.device(device))
		v.upload_image('draw_pixel_variations', img_torch)

if __name__=='__main__':

	device = torch.device('cuda')
	v = viewer('Example')

	# here we can initialize variables that need to be used both by the loop function and the worker threads
	# don't use names that begin with an undescore; these are used by the inner workings of viewer.
	
	v.scale = 1.
	v.message = ''
	
	# Gaussian kernel parameters
	v.angle = 0.0
	v.sx = 1.0
	v.sy = 1.0
	v.size = 15
	v.filter = GaussianFilter(3, v.size, v.sx, v.sy)
	v.kernel = v.filter.weight

	#File and image handling
	v.file_dropdown_current = 0
	v.files = listdir("images/")
	img = np.asarray(Image.open("images/"+v.files[v.file_dropdown_current]))
	v.new_img = False

	v.img = to_tensor(img.copy())
	v.img = v.img.permute(1, 2, 0)
	v.img = v.img.to(torch.device(device))
	v.img_conv = v.img.clone()

	#Middle pixel of image
	v.px = v.img.shape[0]//2
	v.py = v.img.shape[1]//2
	
	# initialize any number of threads
	thread_1 = threading.Thread(target=draw_original, args=(v, device))
	thread_2 = threading.Thread(target=draw_kernel, args=(v, device))
	thread_3 = threading.Thread(target=draw_filtered, args=(v, device))
	thread_4 = threading.Thread(target=draw_pixel_surrounding, args=(v, device))
	thread_5 = threading.Thread(target=draw_multiple_pixel_convolutions, args=(v, device))

	# run! first arg is the loop function, the second is a list/tuple of threads.
	v.start(loop, (thread_1, thread_2, thread_3, thread_4, thread_5))
	# v.start(loop, (thread_1, thread_2, thread_3, thread_4))			# This one excludes the pixel variations windows
