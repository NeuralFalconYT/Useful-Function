# !apt install aria2 -qqy
# pip install tqdm
import requests
import shutil 
import os
import subprocess
from tqdm.auto import tqdm  

def download_huggingface_model_without_HF_TOKEN(repo_id, base_path="/content"):
    """
    In Google Colab, downloading models from Hugging Face can be unnecessarily frustrating.  
    Even when a model is completely open-source and does NOT require a license agreement or token,  
    Colab often forces you to provide a Hugging Face token anyway.  

    That means you have to:  
    1. Go to Hugging Face,  
    2. Generate a Access Tokens, Enter Password,  
    3. Paste it into Colab’s secret keys,  
    4. Restart the runtime,  

    —all just to download a model that should be publicly accessible.  
    It’s a waste of time and breaks the flow of experimentation.  

    This function avoids that hassle by directly fetching the file list via the Hugging Face TOKEN  
    and downloading the files with `aria2c`, no token required (unless the repo truly requires a license).  
    """

    url = f"https://huggingface.co/api/models/{repo_id}"
    download_dir = f"{base_path.rstrip('/')}/{repo_id.split('/')[-1]}"
    
    # reset directory
    if os.path.exists(download_dir):
        shutil.rmtree(download_dir)
    os.makedirs(download_dir, exist_ok=True)

    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        siblings = data.get("siblings", [])
        files = [f["rfilename"] for f in siblings]

        print(f"Found {len(files)} files, downloading into {download_dir} ...")

        # tqdm progress bar
        for file in tqdm(files, desc="Downloading files", unit="file"):
            file_url = f"https://huggingface.co/{repo_id}/resolve/main/{file}"
            cmd = [
                "aria2c", "--console-log-level=error",
                "-c", "-x", "16", "-s", "16", "-k", "1M",
                file_url, "-d", download_dir, "-o", file
            ]
            # print("\nRunning:", " ".join(cmd))
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                tqdm.write(f"❌ Failed: {file}\n{result.stderr}")
    else:
        print("Error:", response.status_code, response.text)


# Example usage
download_huggingface_model_without_HF_TOKEN("Qwen/Qwen-Image-Edit")
