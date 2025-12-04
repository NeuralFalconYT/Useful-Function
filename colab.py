#Auto Restart Colab Runtime
from IPython.display import clear_output
clear_output()
import time
time.sleep(5)
import os
os.kill(os.getpid(), 9)

#@title Keep Alive for Mobile Users
from IPython.display import Audio,display
import numpy as np
display(Audio(np.array([0] * 2 * 3600 * 3000, dtype=np.int8), normalize=False, rate=3000, autoplay=True))

# restart runtime cleanly
import IPython
IPython.Application.instance().kernel.do_shutdown(True)

#play audio
from IPython.display import Audio, display
audio_path = "/content/audio_example.wav"
display(Audio(audio_path))


#free gpu memory
import gc
import torch
del model
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

#gradio ui

import click
@click.command()
@click.option("--debug", is_flag=True, default=False, help="Enable debug mode.")
@click.option("--share", is_flag=True, default=False, help="Enable sharing of the interface.")
def run_demo(share,debug):
    demo1=ui1()
    demo2=ui2()
    demo3=ui3()
    custom_css = """.gradio-container { font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif; }"""
    interface = gr.TabbedInterface([demo1, demo2,demo3],["Page 1","Page 2","Page 3"],title="",theme=gr.themes.Soft(),css=custom_css)
    interface.queue(max_size=10).launch(share=share,debug=debug)
if __name__ == "__main__":
    run_demo()

#find str in a folder
!grep -R "ctc_alignment_mling_uroman_model.pt" -n inference/
