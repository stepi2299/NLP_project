import time

import speech_recognition as sr
import json
import os
import pandas as pd

POSSIBLE_ALGORITHMS = ["google", "sphinx", "vosk", "whisper"]
# sth wrong with amazon and tensorflow, if results will be poor we can try to dealwith these recognitors

r = sr.Recognizer()


def transcribe(algorithm, audio, i=None):
    # recognize speech using Google Speech Recognition
    try:
        if algorithm == "google":
            return r.recognize_google(audio)
        elif algorithm == "sphinx":
            return r.recognize_sphinx(audio)
        # elif algorithm == "amazon":
        #     return r.recognize_amazon(audio)
        # elif algorithm == "tensorflow":
        #     return r.recognize_tensorflow(audio)
        elif algorithm == "whisper":
            return r.recognize_whisper(audio)
        elif algorithm == "vosk":
            return json.loads(r.recognize_vosk(audio))["text"]
    except sr.UnknownValueError:
        print(f"{algorithm} could not understand audio nr: {i}")
        return None
    except sr.RequestError as e:
        print(f"Could not request results from {algorithm} service, sth wrong with dependencies, exception: {e}")
        return None


def speech_2_text(audio_folder_path: str, csv_file_path: str, algorithm: str = "google", denoise: bool = False):
    i = 0
    dict_to_df = {
        "origin": [],
        "heard": [],
        "identical": [],
        "read": [],
        "time_transcription": []
    }
    if algorithm not in POSSIBLE_ALGORITHMS:
        print(f"Given algorithm is not supported, given algorithm: {algorithm}, supported algorithms: {POSSIBLE_ALGORITHMS}")
        return

    csv_file = pd.read_csv(csv_file_path)
    for dialog_nr, utt_nr, sentence in zip(csv_file["Dialogue_ID"], csv_file["Utterance_ID"], csv_file["Utterance"]):
        try:
            audio_file = os.path.join(audio_folder_path, f"dia{dialog_nr}_utt{utt_nr}.wav")
            audio = sr.AudioFile(audio_file)
            with audio as source:
                if denoise:
                    r.adjust_for_ambient_noise(audio)
                audio = r.record(source)  # reading entire audio file
            start = time.perf_counter()
            heard_text = transcribe(algorithm, audio, i)
            dict_to_df["time_transcription"].append(time.perf_counter()-start)
            dict_to_df["origin"].append(sentence)
            dict_to_df["heard"].append(heard_text)
            dict_to_df["identical"].append(heard_text == sentence)
            dict_to_df["read"].append(False if heard_text is None else True)
            if heard_text:
                print(f"DONE index: {i}, sentence: {heard_text}")
        except Exception as e:
            print(f"Sth, went wrong with {i} audio, exception: {e}")
        finally:
            i += 1
    return dict_to_df




