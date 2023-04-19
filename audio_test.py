import speech_recognition as sr
import os
import pandas as pd

audio_folder = "/home/stepi2299/repo/MELD.Raw/audio_dev"

r = sr.Recognizer()
folder_files = os.listdir(audio_folder)
csv_file_path = "/home/stepi2299/repo/MELD.Raw/dev.csv"
csv_file = pd.read_csv(csv_file_path)


def sphinx(audio):
    # recognize speech using Sphinx
    try:
        return r.recognize_sphinx(audio)
    except sr.UnknownValueError:
        print("Sphinx could not understand audio")
        return None
    except sr.RequestError as e:
        print("Sphinx error; {0}".format(e))
        return None


def google_rec(audio):
    # recognize speech using Google Speech Recognition
    try:
        # for testing purposes, we're just using the default API key
        # to use another API key, use `r.recognize_google(audio, key="GOOGLE_SPEECH_RECOGNITION_API_KEY")`
        # instead of `r.recognize_google(audio)`
        return r.recognize_google(audio)
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
        return None
    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition service; {0}".format(e))
        return None

dict_to_df = {
    "origin": [],
    "heard": [],
    "identical": [],
    "read": [],

}

for dialog_nr, utt_nr, sentence in zip(csv_file["Dialogue_ID"], csv_file["Utterance_ID"], csv_file["Utterance"]):
    try:
        audio_file = os.path.join(audio_folder, f"dia{dialog_nr}_utt{utt_nr}.wav")
        audio = sr.AudioFile(audio_file)
        with audio as source:
            audio = r.record(source)  # reading entire audio file
        heard_text = google_rec(audio)
        dict_to_df["origin"].append(sentence)
        dict_to_df["heard"].append(heard_text)
        dict_to_df["identical"].append(heard_text == sentence)
        dict_to_df["read"].append(False if heard_text is None else True)
    except Exception as e:
        print("Sth, went wrong", e)

new_df = pd.DataFrame(dict_to_df)
new_df.to_csv("/home/stepi2299/repo/MELD.Raw/analyze.csv")



