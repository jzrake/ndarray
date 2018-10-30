import struct
import numpy as np


def load(filename):

    with open(filename, 'rb') as f:
        dtype = struct.unpack('8s', f.read(8))[0].decode('utf-8').strip('\x00')
        rank = struct.unpack('i', f.read(4))[0]
        dims = struct.unpack('i' * rank, f.read(4 * rank))
        data = f.read()
        return np.frombuffer(data, dtype=dtype).reshape(dims)


A = load("float64-345.bin")
B = load("int32-88.bin")


print(A.shape, A.dtype)
print(B.shape, B.dtype)
assert((B == np.arange(64).reshape(8, 8)).all())
