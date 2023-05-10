import pytest
import speech_recognition as sr
from src.speech2text.speech_to_text import transcribe, POSSIBLE_ALGORITHMS


@pytest.fixture(name="audio")
def fixture_audio():
    r = sr.Recognizer()
    audio_file = "test_data/test_audio.wav"
    audio = sr.AudioFile(audio_file)
    with audio as source:
        audio = r.record(source)  # reading entire audio file
    return audio


@pytest.mark.parametrize("algorithm", POSSIBLE_ALGORITHMS)
def test_transcribe(audio, algorithm):
    output = transcribe(algorithm, audio)
    assert isinstance(output, str)