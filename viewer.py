
import numpy as np
import torch
import multiprocessing as mp
import threading

import numpy as np
import imgui.core
from imgui.integrations.glfw import GlfwRenderer
import glfw
glfw.ERROR_REPORTING = 'raise'

import ctypes

import OpenGL.GL as gl

import pycuda
import pycuda.gl as cuda_gl
import pycuda.tools

class _texture:
	'''
	This class maps torch tensors to gl textures without a CPU roundtrip.
	'''
	def __init__(self):
		self.tex = gl.glGenTextures(1)
		gl.glBindTexture(gl.GL_TEXTURE_2D, self.tex) # need to bind to modify
		# sets repeat and filtering parameters; change the second value of any tuple to change the value
		for params in ((gl.GL_TEXTURE_WRAP_S, gl.GL_REPEAT), (gl.GL_TEXTURE_WRAP_T, gl.GL_REPEAT), (gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST), (gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)):
			gl.glTexParameteri(gl.GL_TEXTURE_2D, *params)
		self.mapper = None
		self.shape = [0,0]

	# be sure to del textures if you create a forget them often (python doesn't necessarily call del on garbage collect)
	def __del__(self):
		gl.glDeleteTextures(1, [self.tex])
		if self.mapper is not None:
			self.mapper.unregister()

	# the main thing
	def upload(self, image):
		# support for shapes (h,w), (h,w,1), (h,w,3) and (h,w,4)
		if len(image.shape)==2:
			image = image.unsqueeze(-1)
		if image.shape[2] == 1:
			image = image.repeat(1,1,3)
		if image.shape[2]==3:
			image = torch.cat([image, torch.ones_like(image[:,:,0:1])], -1) # cuMemcpy2D works on arrays; they have to be 1,2, or 4-channel

		image = image.to(torch.float32).contiguous() # make the image contiguous in memory
		#image = image.contiguous() # make the image contiguous in memory

		# reallocate if shape changed
		if image.shape[0]!=self.shape[0] or image.shape[1]!=self.shape[1]:
			self.shape = image.shape
			if self.mapper is not None:
				self.mapper.unregister()
			gl.glBindTexture(gl.GL_TEXTURE_2D, self.tex)
			gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA32F, image.shape[1], image.shape[0], 0, gl.GL_RGBA, gl.GL_FLOAT, None)
			gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
			self.mapper = cuda_gl.RegisteredImage(int(self.tex), gl.GL_TEXTURE_2D, pycuda.gl.graphics_map_flags.WRITE_DISCARD)
		
		# map texture to cuda ptr
		tex_data = self.mapper.map()
		tex_arr = tex_data.array(0, 0)

		# copy from torch tensor to mapped gl texture (avoid cpu roundtrip)
		cpy = pycuda.driver.Memcpy2D()
		cpy.set_src_device(image.data_ptr()) # data_ptr() gives cuda pointer; this could be reworked to use any other 
		cpy.set_dst_array(tex_arr)
		cpy.width_in_bytes = cpy.src_pitch = cpy.dst_pitch = 4*image.shape[1]*image.shape[2]
		cpy.height = image.shape[0]
		cpy(aligned=False)

		# cleanup
		tex_data.unmap()
		torch.cuda.synchronize()

class _editable:
	def __init__(self, name, ui_code = '', run_code = ''):
		self.name = name
		# self.ui_code = ui_code if len(ui_code)>0 else 'imgui.begin(\'Test\')\nimgui.text(\'Example\')#your code here!\nimgui.end()'
		self.tentative_ui_code = self.ui_code
		self.run_code = run_code
		self.run_exception = ''
		self.ui_exception = ''
		self.ui_code_visible = False
	def try_execute(self, string, **kwargs):
		try:
			for key, value in kwargs.items():
				locals()[key] = value
			exec(string)
		except Exception as e: # while generally a bad idea, here we truly want to skip any potential error to not disrupt the worker threads
			return 'Exception: ' + str(e)
		return ''
	def loop(self, v):
		imgui.begin(self.name)
		
		self.run_code = imgui.input_text_multiline('run code', self.run_code, 2048)[1]
		if len(self.run_exception)>0:
			imgui.text(self.run_exception)

		_, self.ui_code_visible = imgui.checkbox('Show UI code', self.ui_code_visible)
		if self.ui_code_visible:
			self.tentative_ui_code = imgui.input_text_multiline('ui code', self.tentative_ui_code, 2048)[1]
			if imgui.button('Apply UI code'):
				self.ui_code = self.tentative_ui_code
			if len(self.ui_exception)>0:
				imgui.text(self.ui_exception)
				
		imgui.end()

		self.ui_exception = self.try_execute(self.ui_code, v=v)

	def run(self, **kwargs):
		self.run_exception = self.try_execute(self.run_code, **kwargs)


class viewer:
	def __init__(self, title):
		self.quit = False

		self._images = {}
		self._editables = {}

		glfw.init()
		try:
			with open("viewer.ini", 'r') as file:
				self._width, self._height = (int(i) for i in file.readline().split())
				key = file.readline().rstrip()
				while key is not None and len(key)>0:
					code = [None, None]
					for i in range(2):
						lines = int(file.readline().rstrip())
						code[i] = '\n'.join((file.readline().rstrip() for _ in range(lines)))
					self._editables[key] = _editable(key, code[0], code[1])
					key = file.readline().rstrip()
		except:
			self._width, self._height = 1024, 768

		self._window = glfw.create_window(self._width, self._height, title, None, None)
		glfw.make_context_current(self._window)
		pycuda.driver.init()
		self._cuda_context = pycuda.gl.make_context(pycuda.driver.Device(0))
		glfw.make_context_current(None)

		self._context_lock = mp.Lock()
	
	def _lock(self):
		self._context_lock.acquire()
		try:
			glfw.make_context_current(self._window)
		except Exception as e:
			print(str(e))
			self._context_lock.release()
			return False
		return True

	def _unlock(self):
		glfw.make_context_current(None)
		self._context_lock.release()

	def editable(self, name, **kwargs):
		if name not in self._editables:
			self._editables[name] = _editable(name)
		self._editables[name].run(**kwargs)

	def keydown(self, key):
		return key in self._pressed_keys

	def keyhit(self, key):
		if key in self._hit_keys:
			self._hit_keys.remove(key)
			return True
		return False

	def draw_image(self, name, x_offset = 0, y_offset = 0, scale = 1):
		if name in self._images:
			img = self._images[name]
			imgui.image(img.tex, img.shape[1]*scale, img.shape[0]*scale)

	def start(self, loopfunc, workers = ()):
		# allow single thread object
		if not hasattr(workers, '__len__'):
			workers = (workers,)

		for i in range(len(workers)):
			workers[i].start()

		imgui.create_context()
		self._lock()
		impl = GlfwRenderer(self._window)
		self._unlock()
		
		self._pressed_keys = set()
		self._hit_keys = set()

		def on_key(window, key, scan, pressed, mods):
			if pressed:
				if key not in self._pressed_keys:
					self._hit_keys.add(key)
				self._pressed_keys.add(key)
			else:
				self._pressed_keys.remove(key)
			if key != glfw.KEY_ESCAPE: # imgui erases text with escape (??)
				impl.keyboard_callback(window, key, scan, pressed, mods)

		glfw.set_key_callback(self._window, on_key)
		
		while not (glfw.window_should_close(self._window) or self.keyhit(glfw.KEY_ESCAPE)):
			glfw.poll_events()
			impl.process_inputs()

			self._lock()
			imgui.get_io().display_size = glfw.get_framebuffer_size(self._window)
			imgui.new_frame()
	
			loopfunc(self)

			for key in self._editables:
				self._editables[key].loop(self)

			imgui.render()
			impl.render(imgui.get_draw_data())
			glfw.swap_buffers(self._window)
			gl.glClear(gl.GL_COLOR_BUFFER_BIT)

			self._unlock()
		
		with open("viewer.ini", 'w') as file:
			file.write('{} {}\n'.format(*glfw.get_framebuffer_size(self._window)))
			for k, e in self._editables.items():
				file.write(k+'\n')
				for code in (e.ui_code, e.run_code):
					lines = code.split('\n')
					file.write(str(len(lines))+'\n')
					for line in lines:
						file.write(line+'\n')

		self._lock()
		self.quit = True
		self._unlock()

		for i in range(len(workers)):
			workers[i].join()
			
		glfw.make_context_current(self._window)
		del self._images
		self._images = {}
		glfw.make_context_current(None)
		
		self._cuda_context.pop()

	def upload_image(self, name, image):
		if torch.is_tensor(image):
			if self._lock():
				torch.cuda.synchronize()
				if not self.quit:
					self._cuda_context.push() # set the context for whichever thread wants to upload
					if name not in self._images:
						self._images[name] = _texture()
					self._images[name].upload(image)
					self._cuda_context.pop()
				self._unlock()
