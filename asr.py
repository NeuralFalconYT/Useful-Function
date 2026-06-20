import os
import gc
import time
import requests
import torch
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from faster_whisper import WhisperModel

MODEL_REPO = "deepdml/faster-whisper-large-v3-turbo-ct2"
MODEL_DIR  = "./faster-whisper-large-v3-turbo-ct2"


# ── downloader ────────────────────────────────────────────────────────────────

def _expected_size(url):
    try:
        r = requests.head(url, allow_redirects=True, timeout=10)
        return int(r.headers.get("content-length", -1))
    except Exception:
        return -1

def _download_file(url, path, redownload=False):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    if not redownload and os.path.exists(path):
        local_size = os.path.getsize(path)
        if local_size > 0:
            remote_size = _expected_size(url)
            if remote_size == -1 or local_size == remote_size:
                return "SKIPPED", f"✔️ Skipped: {os.path.basename(path)}"

    try:
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            with open(path, "wb") as f, tqdm(
                total=total, unit="B", unit_scale=True,
                desc=os.path.basename(path), leave=False,
            ) as pbar:
                for chunk in r.iter_content(chunk_size=1024 * 64):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        return "DOWNLOADED", f"⬇️ Downloaded: {os.path.basename(path)}"
    except Exception as e:
        if os.path.exists(path):
            try: os.remove(path)
            except Exception: pass
        return "FAILED", f"❌ Failed: {os.path.basename(path)} ({e})"

def download_model(repo_id, download_folder="./", redownload=False, workers=6):
    start = time.time()
    download_dir = os.path.abspath(download_folder)
    os.makedirs(download_dir, exist_ok=True)
    print(f"📂 {download_dir}")

    # ── parallel download (primary) ──────────────────────────────────────────
    try:
        response = requests.get(f"https://huggingface.co/api/models/{repo_id}", timeout=30)
        response.raise_for_status()
        files = [f["rfilename"] for f in response.json().get("siblings", [])]
        print(f"🚀 Parallel download | {len(files)} files | workers={workers}")

        skipped = downloaded = failed = 0
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {
                ex.submit(
                    _download_file,
                    f"https://huggingface.co/{repo_id}/resolve/main/{file}",
                    os.path.join(download_dir, file),
                    redownload,
                ): file for file in files
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc="Overall"):
                status, msg = future.result()
                if status == "SKIPPED":      skipped += 1
                elif status == "DOWNLOADED": downloaded += 1; print(msg)
                else:                        failed += 1;     print(msg)

        print(f"📊 {downloaded} downloaded | {skipped} skipped | {failed} failed")
        if failed == 0:
            print(f"✅ Done  ⏱ {time.time()-start:.1f}s")
            return download_dir
        print(f"⚠️ {failed} files failed → falling back to snapshot_download")
    except Exception as e:
        print(f"⚠️ Parallel download error ({e}) → falling back to snapshot_download")

    # ── snapshot fallback ─────────────────────────────────────────────────────
    try:
        from huggingface_hub import snapshot_download
        print("🚀 snapshot_download (fallback)...")
        snapshot_download(
            repo_id=repo_id,
            local_dir=download_dir,
            local_dir_use_symlinks=False,
            resume_download=not redownload,
        )
        print(f"✅ Done  ⏱ {time.time()-start:.1f}s")
        return download_dir
    except Exception as e:
        print(f"❌ snapshot_download also failed: {e}")
        return None

LANGUAGE_CODE = {
    'Auto': None,
    'Akan': 'aka', 'Albanian': 'sq', 'Amharic': 'am', 'Arabic': 'ar', 'Armenian': 'hy',
    'Assamese': 'as', 'Azerbaijani': 'az', 'Basque': 'eu', 'Bashkir': 'ba', 'Bengali': 'bn',
    'Bosnian': 'bs', 'Bulgarian': 'bg', 'Burmese': 'my', 'Catalan': 'ca', 'Chinese': 'zh',
    'Croatian': 'hr', 'Czech': 'cs', 'Danish': 'da', 'Dutch': 'nl', 'English': 'en',
    'Estonian': 'et', 'Faroese': 'fo', 'Finnish': 'fi', 'French': 'fr', 'Galician': 'gl',
    'Georgian': 'ka', 'German': 'de', 'Greek': 'el', 'Gujarati': 'gu', 'Haitian Creole': 'ht',
    'Hausa': 'ha', 'Hebrew': 'he', 'Hindi': 'hi', 'Hungarian': 'hu', 'Icelandic': 'is',
    'Indonesian': 'id', 'Italian': 'it', 'Japanese': 'ja', 'Kannada': 'kn', 'Kazakh': 'kk',
    'Korean': 'ko', 'Kurdish': 'ckb', 'Kyrgyz': 'ky', 'Lao': 'lo', 'Lithuanian': 'lt',
    'Luxembourgish': 'lb', 'Macedonian': 'mk', 'Malay': 'ms', 'Malayalam': 'ml', 'Maltese': 'mt',
    'Maori': 'mi', 'Marathi': 'mr', 'Mongolian': 'mn', 'Nepali': 'ne', 'Norwegian': 'no',
    'Norwegian Nynorsk': 'nn', 'Pashto': 'ps', 'Persian': 'fa', 'Polish': 'pl', 'Portuguese': 'pt',
    'Punjabi': 'pa', 'Romanian': 'ro', 'Russian': 'ru', 'Serbian': 'sr', 'Sinhala': 'si',
    'Slovak': 'sk', 'Slovenian': 'sl', 'Somali': 'so', 'Spanish': 'es', 'Sundanese': 'su',
    'Swahili': 'sw', 'Swedish': 'sv', 'Tamil': 'ta', 'Telugu': 'te', 'Thai': 'th',
    'Turkish': 'tr', 'Ukrainian': 'uk', 'Urdu': 'ur', 'Uzbek': 'uz', 'Vietnamese': 'vi',
    'Welsh': 'cy', 'Yiddish': 'yi', 'Yoruba': 'yo', 'Zulu': 'zu'
}
CODE_TO_NAME = {v: k for k, v in LANGUAGE_CODE.items() if v}

# ── model singleton ───────────────────────────────────────────────────────────

_model = None

def load_model():
    global _model
    if _model is not None:
        return _model

    device       = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if torch.cuda.is_available() else "int8"

    model_path = MODEL_DIR if os.path.isdir(MODEL_DIR) else \
                 download_model(MODEL_REPO, download_folder="./", redownload=False)

    _model = WhisperModel(model_path, device=device, compute_type=compute_type)
    return _model


def unload_model():
    global _model
    _model = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ── main function ─────────────────────────────────────────────────────────────

def get_transcript(audio_path: str, language: str = "Auto") -> tuple[str, str]:
    """
    Transcribe audio and return (transcript_text, detected_language_name).

    Args:
        audio_path : path to audio/video file
        language   : language name from LANGUAGE_CODE keys, or "Auto"

    Returns:
        (transcript_text, language_name)
        e.g. ("Hello world ...", "English")
    """
    model    = load_model()
    lang_code = LANGUAGE_CODE.get(language)          # None → auto-detect

    kwargs = dict(word_timestamps=False)
    if lang_code:
        kwargs["language"] = lang_code

    segments, info = model.transcribe(audio_path, **kwargs)

    text          = " ".join(s.text.strip() for s in segments)
    detected_name = CODE_TO_NAME.get(info.language, info.language)

    unload_model()
    return text, detected_name


# ── usage ─────────────────────────────────────────────────────────────────────
# from asr import get_transcript
#
# text, lang = get_transcript("/content/audio.mp3")           # auto-detect
# text, lang = get_transcript("/content/audio.mp3", "Hindi")  # force language
# print(lang, text)
