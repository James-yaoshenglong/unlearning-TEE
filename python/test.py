import sys
sys.path.append('./datasets/purchase')


import ctypes
from ctypes import *
import numpy as np 
from numpy.ctypeslib import ndpointer 
import importlib
import dataloader
import time

floatp = ndpointer(dtype=np.float32, ndim=1, flags="CONTIGUOUS") 

ll = ctypes.cdll.LoadLibrary
lib = ll("./App/app.so")

lib.initialize_enclave.restype = c_uint32
lib.initialize_enclave.argtypes = []

# lib.destroy_enclave.restype = c_void
lib.destroy_enclave.argtypes = []

lib.load_data.argtypes = [floatp, floatp, c_uint32, c_uint32]
lib.xxhash.argtypes = [floatp, c_uint32]
lib.xxhash.restype = c_uint64

lib.unlearning.argtypes = [c_uint64]

lib.predict.argtypes = [floatp, floatp, c_uint32]

# lib.cnn_inference_f32_cpp.restype = c_int32

split = np.load("./containers/default/splitfile.npy", allow_pickle=True)
data, label = dataloader.load(split[0])

# data = np.arange(9., dtype=np.float32)
# label = np.zeros(3, dtype = np.float32, order = 'C')

r, c = data.shape

data = data.astype(np.float32)

# unlearning_set = [1, 30024, 54567]
unlearning_set = [30024]
unlearning_ids = []

for i in unlearning_set:
    temp = data[i]
    temp = np.append(temp, label[i].astype(np.float32))
    print(lib.xxhash(temp, (c+1)*4))
    unlearning_ids.append(lib.xxhash(temp, (c+1)*4))

data = np.reshape(data, (-1,))
label = label.astype(np.float32)

s = lib.initialize_enclave()

print(data.shape)
print(label.shape)

lib.load_data(data, label, r, c)

start = time.time()

lib.init_enclave_storage()
tick = time.time()
print("training need time", tick-start)

data, label = dataloader.load([x for x in range(31152)])
data = data.astype(np.float32)
data = np.reshape(data, (-1,))
label = label.astype(np.float32)
lib.predict(data, label, 31152)



for id in unlearning_ids:
    tick = time.time()
    lib.unlearning(id)
    print("unlearning need time", time.time()-tick)
    lib.predict(data, label, 31152)

# # lib.cnn_inference_f32_cpp()

lib.destroy_enclave()