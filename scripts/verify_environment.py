
import sys
import importlib.util

print("=" * 45)
print("NEO Detection - Environment Check")
print("=" * 45)

# 1. Check the Python interpreter
print(f"\nPython version: {sys.version.split()[0]}")
print(f"Python executable: {sys.executable}")

# 2. Check core libraries
libraries = [
    "torch",
    "ultralytics",
    "numpy",
    "cv2",
    "PIL",
    "matplotlib",
    "yaml",
]

print("\nLibrary checks:")

for library in libraries:
    installed = importlib.util.find_spec(library) is not None
    status = "OK" if installed else "MISSING"
    print(f"  {library}: {status}")

# 3. Check PyTorch and GPU availability
print("\nGPU checks:")

if importlib.util.find_spec("torch") is not None:
    import torch

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA build: {torch.version.cuda}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

        # Perform a small calculation on the GPU
        x = torch.randn(512, 512, device="cuda")
        y = torch.randn(512, 512, device="cuda")
        result = x @ y
        torch.cuda.synchronize()

        print(f"Test tensor device: {result.device}")
        print(f"Test tensor shape: {tuple(result.shape)}")
        print("GPU calculation: PASSED")
    else:
        print("GPU calculation: NOT RUN")
        print("Check your NVIDIA driver and PyTorch CUDA installation.")

# 4. Check whether the dataset folder exists
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
dataset_path = project_root / "NEOBench"

print("\nDataset check:")
print(f"Dataset folder exists: {dataset_path.exists()}")
print(f"Dataset path: {dataset_path}")

print("\nEnvironment check complete.")
