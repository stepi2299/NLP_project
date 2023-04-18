import speech_recognition as sr
import moviepy.editor as mp
from pathlib import Path
import os

media_folder = "/home/stepi2299/repo/MELD.Raw/dev"
dest_audio_folder = "/home/stepi2299/repo/MELD.Raw/audio_dev"

Path(dest_audio_folder).mkdir(parents=True, exist_ok=True)
media_files = os.listdir(media_folder)
# media_files.sort()

for i, file in enumerate(media_files):
    media_file = os.path.join(media_folder, file)
    print(f"Saving file: {file}")
    name = file.split(".")[0]
    dest_name = f"{name}.wav"
    dest_file = os.path.join(dest_audio_folder, dest_name)
    clip = mp.VideoFileClip(media_file)
    clip.audio.write_audiofile(dest_file)
    print(f"Already saved file nr {dest_file}")


