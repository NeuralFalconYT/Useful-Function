import subprocess
import torch
import gc
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def get_max_gpu_memory():
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory
        return round(total_memory / (1024 ** 3), 2)  # Convert to GB
    else:
        print("CUDA is not available.")
        return None

def is_gpu_memory_over_limit(safety_margin_gb=0.6):
    """
    Returns True if GPU memory usage exceeds (total - safety_margin).
    """
    max_memory_gb = get_max_gpu_memory()
    if max_memory_gb is None:
        return False  # Can't check memory if no GPU

    limit_gb = max_memory_gb - safety_margin_gb
    print(f"GPU memory limit set to {limit_gb} GB.")
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        memory_used_mb_list = result.stdout.strip().splitlines()
        for i, memory_used_mb in enumerate(memory_used_mb_list):
            memory_used_gb = int(memory_used_mb) / 1024.0
            if memory_used_gb > limit_gb:
                print("⚠️ GPU memory is over the safe limit. Avoid loading large models.")
                return True
        print("✅ GPU memory is within safe limits.")
        return False
    except Exception as e:
        print(f"Failed to check GPU memory: {e}")
        return False


model=None
def load_model():
    global model
    if model is None:
      del model 
      gc.collect()
      torch.cuda.empty_cache()
    model = your_torch_model(DEVICE)
    return model

#initialize load model for the first time 
chatterbox_model=load_model()

#for multiple generation on low gpu memory use this logic, you will not get any error cuda out of memory 
if is_gpu_memory_over_limit():
      model=load_model()
