root_path = "/content"

from hf_mirror import download_model
whisper_model_path=download_model(
    "deepdml/faster-whisper-large-v3-turbo-ct2",
    download_folder="./faster-whisper-large-v3-turbo-ct2",
    redownload=True,
    workers=6,
    use_snapshot=True,  
)
import torch
import gc
from faster_whisper import WhisperModel

LANGUAGE_CODE = {
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

def get_language_name(code):
    """Retrieves the full language name from its code."""
    for name, value in LANGUAGE_CODE.items():
        if value == code:
            return name
    return None
def transcribe_audio(audio_path, language=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    lang_code = LANGUAGE_CODE.get(language)
    whisper_path=f"{root_path}/fish-speech-colab/faster-whisper-large-v3-turbo-ct2"
    model = WhisperModel(
        whisper_path,
        device=device,
        compute_type=compute_type
    )
    segments, info = model.transcribe(audio_path, language=lang_code)
    detected_lang_code = info.language
    detected_language = get_language_name(detected_lang_code)
    transcript = " ".join([s.text for s in segments])

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return transcript.strip(),detected_language

audio_path="audio.mp3"
transcript,detected_language=transcribe_audio(audio_path, language="English")
transcript    
