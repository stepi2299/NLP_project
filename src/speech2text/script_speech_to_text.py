from speech_to_text import speech_2_text
import pandas as pd

audio_folder = "/home/stepi2299/repo/data/MELD.Raw/audio_dev"
csv_file_path = "/home/stepi2299/repo/data/MELD.Raw/dev.csv"
destination_file = "/home/stepi2299/repo/NLP_project/csv_files"
ALGORITHM = "whisper"


output = speech_2_text(audio_folder_path=audio_folder, csv_file_path=csv_file_path, algorithm=ALGORITHM, denoise=False)

new_df = pd.DataFrame(output)
dest_path_heard_test = f"{destination_file}/{ALGORITHM}.csv"
new_df.to_csv(dest_path_heard_test, index=False)
