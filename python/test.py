import ctypes
from ctypes import *
import numpy as np 
from numpy.ctypeslib import ndpointer 

floatp = ndpointer(dtype=np.float32, ndim=1, flags="CONTIGUOUS") 

ll = ctypes.cdll.LoadLibrary
lib = ll("./App/app.so")

lib.initialize_enclave.restype = c_uint32
lib.initialize_enclave.argtypes = []

# lib.destroy_enclave.restype = c_void
lib.destroy_enclave.argtypes = []

lib.load_data.argtypes = [floatp, floatp, c_uint32, c_uint32]

# lib.cnn_inference_f32_cpp.restype = c_int32



arr = np.arange(9., dtype=np.float32)
label = np.zeros(3, dtype = np.float32, order = 'C')


s = lib.initialize_enclave()

print(arr)
print(label)

lib.load_data(arr, label, 3, 3)


lib.init_enclave_storage()

# lib.cnn_inference_f32_cpp()

lib.destroy_enclave()