#Auto Restart Colab Runtime
from IPython.display import clear_output
clear_output()
import time
time.sleep(5)
import os
os.kill(os.getpid(), 9)
#or 




from IPython.display import Audio,display
from IPython.display import clear_output
import time
clear_output()

display(Audio("https://raw.githubusercontent.com/NeuralFalconYT/Useful-Function/refs/heads/main/audio/warning.mp3", autoplay=True))
time.sleep(6)
clear_output()
time.sleep(5)
import os
os.kill(os.getpid(), 9)



#@title Keep Alive for Mobile Users
from IPython.display import Audio,display
import numpy as np
display(Audio(np.array([0] * 2 * 3600 * 3000, dtype=np.int8), normalize=False, rate=3000, autoplay=True))
 # or
#@title Prevent Colab Sleep (If You Are Running Colab on a Smartphone)
from IPython.display import Audio,display
display(Audio("https://raw.githubusercontent.com/KoboldAI/KoboldAI-Client/main/colab/silence.m4a", autoplay=True))
 # or
%%html
<b>Press play on the music player to keep the tab alive, then run the cell below</b><br/>
<audio src="https://raw.githubusercontent.com/KoboldAI/KoboldAI-Client/main/colab/silence.m4a" controls>

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


def ui():
  custom_css = """.gradio-container { font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif; }"""

  with gr.Blocks(theme=gr.themes.Soft(), css=custom_css) as demo:
      gr.HTML("""
          <div style="text-align: center; margin: 20px auto; max-width: 800px;">
              <h1 style="font-size: 2.5em; margin-bottom: 5px;">🎙️ VibeVoice-Realtime-0.5B </h1>
          </div>""")


          with gr.Accordion("Examples", open=True):
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





# Check which Python libraries are installed and which need to be installed or upgraded
import pkg_resources
from packaging import version

requirements = {
    "setuptools": "80.9.0",
    "diffusers": "0.35.2",
    "transformers": "4.57.3",
    "accelerate": "1.12.0",
}

installed = {pkg.key: pkg.version for pkg in pkg_resources.working_set}

for pkg, req_version in requirements.items():
    if pkg in installed:
        ok = version.parse(installed[pkg]) >= version.parse(req_version)
        status = "✅ OK" if ok else "⚠️ Too old"
        print(f"{pkg:20} installed: {installed[pkg]:10} required: {req_version:10} {status}")
    else:
        print(f"{pkg:20} ❌ NOT INSTALLED")
