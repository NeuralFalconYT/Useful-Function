# Useful function for any TTS 
from sentence_splitter import SentenceSplitter
import re
import uuid
import numpy as np
import wave
from pydub import AudioSegment
from pydub.silence import split_on_silence

def word_split(text, char_limit=300):
    words = text.split()
    chunks = []
    current_chunk = ""

    for word in words:
        if len(current_chunk) + len(word) + (1 if current_chunk else 0) <= char_limit:
            current_chunk += (" " if current_chunk else "") + word
        else:
            chunks.append(current_chunk)
            current_chunk = word

    if current_chunk:
        chunks.append(current_chunk)

    return chunks

def split_into_chunks(text, max_char_limit=300):
    splitter = SentenceSplitter(language='en')
    raw_sentences = splitter.split(text)

    # Flattened list of sentence-level word chunks
    sentence_chunks = []
    for sen in raw_sentences:
        sentence_chunks.extend(word_split(sen, char_limit=max_char_limit))

    chunks = []
    temp_str = ""

    for sentence in sentence_chunks:
        if len(temp_str) + len(sentence) + (1 if temp_str else 0) <= max_char_limit:
            temp_str += (" " if temp_str else "") + sentence
        else:
            chunks.append(temp_str)
            temp_str = sentence

    if temp_str:
        chunks.append(temp_str)

    return chunks


def split_text_into_chunks(text, chunk_size=400):
    """
    Split long text into smaller chunks of max length `chunk_size`.
    """
    # Split by punctuation followed by space (preserves sentence boundaries)
    sentences = re.split(r'(?<=[.!?]) +', text)
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) > chunk_size:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
        current_chunk += sentence + " "
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks
    
def clean_text(text):
    # Define replacement rules
    replacements = {
        "–": " ",  # Replace en-dash with space
        "—": " ",  # 
        "-": " ",  # Replace hyphen with space
        "**": " ", # Replace double asterisks with space
        "*": " ",  # Replace single asterisk with space
        "#": " ",  # Replace hash with space
    }

    # Apply replacements
    for old, new in replacements.items():
        text = text.replace(old, new)

    # Remove emojis using regex (covering wide range of Unicode characters)
    emoji_pattern = re.compile(
        r'[\U0001F600-\U0001F64F]|'  # Emoticons
        r'[\U0001F300-\U0001F5FF]|'  # Miscellaneous symbols and pictographs
        r'[\U0001F680-\U0001F6FF]|'  # Transport and map symbols
        r'[\U0001F700-\U0001F77F]|'  # Alchemical symbols
        r'[\U0001F780-\U0001F7FF]|'  # Geometric shapes extended
        r'[\U0001F800-\U0001F8FF]|'  # Supplemental arrows-C
        r'[\U0001F900-\U0001F9FF]|'  # Supplemental symbols and pictographs
        r'[\U0001FA00-\U0001FA6F]|'  # Chess symbols
        r'[\U0001FA70-\U0001FAFF]|'  # Symbols and pictographs extended-A
        r'[\U00002702-\U000027B0]|'  # Dingbats
        r'[\U0001F1E0-\U0001F1FF]'   # Flags (iOS)
        r'', flags=re.UNICODE)
  
    text = emoji_pattern.sub(r'', text)

    # Remove multiple spaces and extra line breaks
    text = re.sub(r'\s+', ' ', text).strip()

    return text




def tts_file_name(text, language="en"):
    # Clean and process the text
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Keep only letters and spaces
    text = text.lower().strip().replace(" ", "_")

    # Ensure the text is not empty
    if not text:
        text = "audio"

    # Truncate to first 20 characters for filename
    truncated_text = text[:20]

    # Sanitize and format the language tag
    language = re.sub(r'\s+', '_', language.strip().lower()) if language else "unknown"

    # Generate random suffix
    random_string = uuid.uuid4().hex[:8].upper()

    # Construct the filename
    file_name = f"./{truncated_text}_{language}_{random_string}.wav"
    return file_name








def remove_silence_function(file_path,minimum_silence=50):
    # Extract file name and format from the provided path
    output_path = file_path.replace(".wav", "_no_silence.wav")
    audio_format = "wav"
    # Reading and splitting the audio file into chunks
    sound = AudioSegment.from_file(file_path, format=audio_format)
    audio_chunks = split_on_silence(sound,
                                    min_silence_len=100,
                                    silence_thresh=-45,
                                    keep_silence=minimum_silence) 
    # Putting the file back together
    combined = AudioSegment.empty()
    for chunk in audio_chunks:
        combined += chunk
    combined.export(output_path, format=audio_format)
    return output_path
