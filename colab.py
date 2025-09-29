#Auto Restart Colab Runtime
from IPython.display import clear_output
clear_output()
import time
time.sleep(5)
import os
os.kill(os.getpid(), 9)


#play audio
from IPython.display import Audio, display
audio_path = "/content/audio_example.wav"
display(Audio(audio_path))
