#Auto Restart Colab Runtime
from IPython.display import clear_output
clear_output()
import time
time.sleep(5)
import os
os.kill(os.getpid(), 9)
