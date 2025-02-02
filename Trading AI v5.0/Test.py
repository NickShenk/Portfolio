import ctypes

mylib = ctypes.CDLL("./Fitness.so")
mylib.think.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_int), ctypes.c_int]
mylib.think.restype = ctypes.POINTER(ctypes.c_float)
def newthink(inp,weights,shape):
   # Define argument and return types (if needed)
   return mylib.think((ctypes.c_float * len(inp))(*inp),len(inp),(ctypes.c_float * len(weights))(*weights),(ctypes.c_int * len(shape))(*shape),len(shape))



def think(previous,weights,shape):
   arr1 = [0.0 for _ in range(max(shape))]
   arr1[:len(previous)] = previous
   arr2 = [0.0 for _ in range(max(shape))]
   offset = 0
   lastm = shape[0]
   for m in shape[1:]:
      for x in range(m):
         summation = weights[offset]
         for y in range(lastm):
            summation += weights[offset + y + 1] * arr1[y]
         offset += 1 + lastm
         arr2[x] = max(summation,summation*.1)
      arr1, arr2 = arr2, arr1 # flip arrays to avoid reinitializing
      lastm = m
   return [(x) for x in arr1[:shape[m]]]




import random
shape = [5,32,32,3]
weights = [random.randrange(3) - 1 for _ in range(sum(shape[1:]) + sum([shape[m-1]*shape[m] for m in range(1,len(shape))]))]
inp = [random.randrange(3) - 1 for _ in range(shape[0])]
import time
runs = 1
start = time.time()
for _ in range(runs):
    print(think(inp,weights,shape))
print("TIME TAKEN1 :",time.time()-start)
print("STARTED")
start = time.time()
for _ in range(runs):
    out =  newthink(inp,weights,shape)
    print([out[x] for x in range(shape[-1])])
print("TIME TAKEN2 :",time.time()-start)
