nsys profile --trace=cuda --sample=none --cpuctxsw=none --force-overwrite=true -o nsys_temp ../cmake-build-debug/d_core_cuda.exe

nsys stats --force-export=true -r cuda_gpu_sum nsys_temp.nsys-rep