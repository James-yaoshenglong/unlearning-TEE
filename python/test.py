import ctypes
from ctypes import *
import numpy as np 
from numpy.ctypeslib import ndpointer 

doublep = ndpointer(dtype=np.double, ndim=1, flags="CONTIGUOUS") 

ll = ctypes.cdll.LoadLibrary
lib = ll("./App/app.so")

lib.initialize_enclave.restype = c_uint32
lib.initialize_enclave.argtypes = []

# lib.destroy_enclave.restype = c_void
lib.destroy_enclave.argtypes = []

lib.load_data.argtypes = [doublep, c_uint32, c_uint32]

# lib.cnn_inference_f32_cpp.restype = c_int32



arr = np.arange(9., dtype=np.float64)


s = lib.initialize_enclave()

print(arr)

lib.load_data(arr, 3, 3)

lib.init_enclave_storage()

# lib.cnn_inference_f32_cpp()

lib.destroy_enclave()