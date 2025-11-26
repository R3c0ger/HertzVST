import sys

import torch


def test_basic_cuda():
    """Test basic CUDA functionality"""
    print("-"*30)
    print("Basic CUDA Functionality Test: ")
    print("-"*30)
    print(f"Python Version: {sys.version}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"PyTorch Path: {torch.__file__}")
    is_cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {is_cuda_available}")
    if not is_cuda_available:
        print("❌ ERROR: CUDA not available!")
        print("This indicates PyTorch was installed without CUDA support")
        return False

    print(f"CUDA Version: {torch.version.cuda}")
    print(f"cuDNN Version: {torch.backends.cudnn.version()}")
    print(f"GPU Count: {torch.cuda.device_count()}")

    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  - Compute Capability: {torch.cuda.get_device_capability(i)}")
        print(f"  - Memory: {torch.cuda.get_device_properties(i).total_memory / 1024 ** 3:.1f} GB")

    return True


if __name__ == "__main__":
    success = test_basic_cuda()
    if success:
        print("✅ Basic CUDA functionality test passed!")
    else:
        print("❌ Basic CUDA functionality test failed!")
