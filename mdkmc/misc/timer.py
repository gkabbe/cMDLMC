import time

class TimeIt:
	def __init__(self, f):
		self.f = f
	def __call__(self, *args):
		start_time = time.time()
		self.f(*args)
		print("#Time:", time.time() - start_time)
