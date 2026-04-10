import cupy as cp

# 1. Check the total number of GPUs CuPy can detect
num_gpus = cp.cuda.runtime.getDeviceCount()
print(f"Total GPUs detected by CuPy: {num_gpus}")

if num_gpus > 0:
    # 2. Get the currently active device object
    current_device = cp.cuda.Device()
    
    # 3. Determine its device number (ID)
    print(f"Currently active GPU ID: {current_device.id}")
    
    # Optional: Get the compute capability of the GPU
    print(f"Compute Capability: {current_device.compute_capability}")
else:
    print("CuPy cannot detect any GPUs.")