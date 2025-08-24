# !apt install aria2 -qqy
# !pip install tqdm

import requests, os, subprocess
from tqdm.auto import tqdm
def download_huggingface_model_without_HF_TOKEN(repo_id, download_folder="./", redownload=False):
    """
    In Google Colab, downloading models from Hugging Face can be unnecessarily frustrating.  
    Even when a model is completely open-source and does NOT require a license agreement or token,  
    Colab often forces you to provide a Hugging Face token anyway.  

    That means you have to:  
    1. Go to Hugging Face,  
    2. Generate a Access Tokens, Enter Password,  
    3. Paste it into Colab’s secret keys,  
    4. Restart the runtime,  

    all just to download a model that should be publicly accessible.  
    It’s a waste of time and breaks the flow of experimentation.  

    This function avoids that hassle by directly fetching the file list via the Hugging Face TOKEN  
    and downloading the files with `aria2c`, no token required (unless the repo truly requires a license).  
    """
    url = f"https://huggingface.co/api/models/{repo_id}"
    download_dir = os.path.abspath(f"{download_folder.rstrip('/')}/{repo_id.split('/')[-1]}")
    os.makedirs(download_dir, exist_ok=True)

    print(f"📂 Download directory: {download_dir}")

    response = requests.get(url)
    if response.status_code != 200:
        print("❌ Error:", response.status_code, response.text)
        return

    data = response.json()
    siblings = data.get("siblings", [])
    files = [f["rfilename"] for f in siblings]

    print(f"📦 Found {len(files)} files in repo '{repo_id}'. Checking cache ...")

    for file in tqdm(files, desc="Downloading files", unit="file"):
        file_path = os.path.join(download_dir, file)

        # ✅ If file exists
        if os.path.exists(file_path):
            if redownload:
                # delete before re-download
                os.remove(file_path)
                tqdm.write(f"♻️ Redownloading: {file}")
            elif os.path.getsize(file_path) > 0:
                tqdm.write(f"✔️ Skipped (already exists): {file}")
                continue  # skip good file

        # make sure parent directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # build download command
        file_url = f"https://huggingface.co/{repo_id}/resolve/main/{file}"
        cmd = [
            "aria2c", "--console-log-level=error",
            "-c", "-x", "16", "-s", "16", "-k", "1M",
            file_url, "-d", download_dir, "-o", file
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            tqdm.write(f"❌ Failed: {file}\n{result.stderr}")
        else:
            tqdm.write(f"⬇️ Downloaded: {file}")
    return download_dir            
# download_huggingface_model_without_HF_TOKEN("deepdml/faster-whisper-large-v3-turbo-ct2", download_folder="./", redownload=False)
