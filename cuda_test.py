import torch
import time
import psutil
import os
import GPUtil
from pynvml import *

def get_memory_usage():
    """Get current memory usage for CPU and GPU."""
    # CPU Memory
    process = psutil.Process(os.getpid())
    cpu_memory = process.memory_info().rss / 1024 / 1024  # Convert to MB
    
    # GPU Memory
    gpu_memory = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpu_memory.append({
                'device': i,
                'memory_allocated': torch.cuda.memory_allocated(i) / 1024 / 1024,  # MB
                'memory_cached': torch.cuda.memory_reserved(i) / 1024 / 1024  # MB
            })
    
    return cpu_memory, gpu_memory

def test_cuda_availability():
    """Test CUDA availability and print device information."""
    print("\n=== CUDA Availability Test ===")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of devices: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(0)}")

def benchmark_matrix_operations(size=5000, iterations=10):
    """Benchmark matrix operations on CPU and GPU."""
    print("\n=== Matrix Operations Benchmark ===")
    
    # Generate random matrices
    matrix_a = torch.randn(size, size)
    matrix_b = torch.randn(size, size)
    
    # CPU benchmark
    start_time = time.time()
    for _ in range(iterations):
        result_cpu = torch.mm(matrix_a, matrix_b)
    cpu_time = (time.time() - start_time) / iterations
    print(f"CPU average time: {cpu_time:.4f} seconds")
    
    # GPU benchmark
    if torch.cuda.is_available():
        matrix_a_gpu = matrix_a.cuda()
        matrix_b_gpu = matrix_b.cuda()
        
        # Warm-up
        torch.mm(matrix_a_gpu, matrix_b_gpu)
        torch.cuda.synchronize()
        
        start_time = time.time()
        for _ in range(iterations):
            result_gpu = torch.mm(matrix_a_gpu, matrix_b_gpu)
            torch.cuda.synchronize()
        gpu_time = (time.time() - start_time) / iterations
        print(f"GPU average time: {gpu_time:.4f} seconds")
        print(f"GPU speedup: {cpu_time/gpu_time:.2f}x")

def monitor_gpu_utilization():
    """Monitor GPU utilization using GPUtil."""
    print("\n=== GPU Utilization ===")
    try:
        nvmlInit()
        gpus = GPUtil.getGPUs()
        for gpu in gpus:
            print(f"\nGPU ID: {gpu.id}")
            print(f"GPU Load: {gpu.load*100:.1f}%")
            print(f"GPU Memory Used: {gpu.memoryUsed}MB")
            print(f"GPU Memory Total: {gpu.memoryTotal}MB")
            print(f"GPU Temperature: {gpu.temperature}°C")
    except Exception as e:
        print(f"Error monitoring GPU: {e}")

def main():
    print("Starting CUDA utilization tests...")
    
    # Test CUDA availability
    test_cuda_availability()
    
    # Print initial memory usage
    print("\n=== Initial Memory Usage ===")
    cpu_mem, gpu_mem = get_memory_usage()
    print(f"CPU Memory Usage: {cpu_mem:.2f} MB")
    for gpu in gpu_mem:
        print(f"GPU {gpu['device']} Allocated Memory: {gpu['memory_allocated']:.2f} MB")
        print(f"GPU {gpu['device']} Cached Memory: {gpu['memory_cached']:.2f} MB")
    
    # Run benchmark
    benchmark_matrix_operations()
    
    # Monitor GPU utilization
    monitor_gpu_utilization()
    
    # Print final memory usage
    print("\n=== Final Memory Usage ===")
    cpu_mem, gpu_mem = get_memory_usage()
    print(f"CPU Memory Usage: {cpu_mem:.2f} MB")
    for gpu in gpu_mem:
        print(f"GPU {gpu['device']} Allocated Memory: {gpu['memory_allocated']:.2f} MB")
        print(f"GPU {gpu['device']} Cached Memory: {gpu['memory_cached']:.2f} MB")

if __name__ == "__main__":
    main() 