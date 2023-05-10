# NLP_project

## Configuration
1. Create virtual environment using virtualvenv, venv or conda
2. Activate your environment
3. Install dependencies by pasting

   ```sh
   pip install -r requirements.txt
   ```

4. To use *whisper* as speech to text tool install openai-whisper

   ```sh
   python3 -m pip install git+https://github.com/openai/whisper.git soundfile
   ```

## Speech Recognition Engines Comparison

<p> 
To conduct further research we have to choose speech recognition engine. 
Library Speech Recognition provide many engines with easy setup. We are interested in free solutions. 
</p>

### Dataset
MELD - https://affective-meld.github.io/
Engines will be compared using 1108 sentences from serial *Friends*

Example sentences:

| Origin Transcription                           |
|------------------------------------------------|
| Oh my God, he's lost it. He's totally lost it. | 
| Now, there's two reasons.                      |
| Yes? Yes?! How can I help you?                 | 
| I'm sorry, who are you?                        | 
| Okay. You ready to push again?                 | 
| Noooo!!                                        |

### Comparison

1. Google Recognition

   |                                                             | count | percentage |
   |-------------------------------------------------------------|-------|------------|
   | transcription accomplished                                  | 729   | 65.79%     |
   | identical transcription                                     | 0     | 0%         |
   | identical words in transcription <br/>(without punctuation) | 97    | 8.75%      |
   
   Percentage of correct Words in transcription: 44.78%
   
   |      | Processing Time |
   |------|-----------------|
   | min  | 0.08s           |
   | mean | 0.79s           |
   | max  | 4.57s           |

2. Sphinx

   |                                                             | count | percentage |
   |-------------------------------------------------------------|-------|------------|
   | transcription accomplished                                  | 1088  | 98.19%     |
   | identical transcription                                     | 2     | 0.18%      |
   | identical words in transcription <br/>(without punctuation) | 32    | 2.89%      |
   
   Percentage of correct Words in transcription: 20.28%
   
   |      | Processing Time |
   |------|-----------------|
   | min  | 0.36s           |
   | mean | 3.28s           |
   | max  | 23.54s          |
3. Vosk

   |                                                             | count | percentage |
   |-------------------------------------------------------------|-------|------------|
   | transcription accomplished                                  | 1023  | 92.33%     |
   | identical transcription                                     | 1     | 0.09%      |
   | identical words in transcription <br/>(without punctuation) | 124   | 11.19%     |
   
   Percentage of correct Words in transcription: 44.83%
   
   |      | Processing Time |
   |------|-----------------|
   | min  | 0.17s           |
   | mean | 0.97s           |
   | max  | 3.1s            |

4. Whisper

   |                                                             | count | percentage |
   |-------------------------------------------------------------|-------|------------|
   | transcription accomplished                                  | 1089  | 98.29%     |
   | identical transcription                                     | 0     | 0%         |
   | identical words in transcription <br/>(without punctuation) | 280   | 25.27%     |
   
   Percentage of correct Words in transcription: 54.26%
   
   |      | Processing Time |
   |------|-----------------|
   | min  | 2.45s           |
   | mean | 3.5s            |
   | max  | 35.75s          |

### Conclusion

Engine delivered by OpenAI named whisper is the best in almost all statistics (except identical transcription but it is extremely hard to transcript all punctuations etc.)
On the other hand time of processing is very big what can be problematic.
