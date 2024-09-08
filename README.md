# lmstudio_offline_voice_assistant_v2.0
 A fully offline voice assistant that combines lmstudio and applio together. Uses two methods of TTS, STT and also has some extra features. 

Massive improvements over v1. Whisper text to speech mode added. Direct communication to the applio API finally! chat history. GUI improvements and more! Hope you have fun with it!

https://github.com/MushieKings
https://www.youtube.com/@mushiekings/videos
https://paypal.me/MushieKingz

Should work on most python versions. I tested on 3.10 and 3.11 just fine.
Might need cmake to build some wheels. Be sure to add to PATH during installation. Also might need some visual studio build tools if that doesn't work.
https://cmake.org/download/

Make sure Applio is up to date.

-You need to download vosk-model-small-en-us-0.15 or better and extract in root folder
https://alphacephei.com/vosk/models

Whisper Model Download Links
    "tiny.en": "https://openaipublic.azureedge.net/main/whisper/models/d3dd57d32accea0b295c96e26691aa14d8822fac7d9d27d5dc00b4ca2826dd03/tiny.en.pt",
    "tiny": "https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt",
    "base.en": "https://openaipublic.azureedge.net/main/whisper/models/25a8566e1d0c1e2231d1c762132cd20e0f96a85d16145c3a00adf5d1ac670ead/base.en.pt",
    "base": "https://openaipublic.azureedge.net/main/whisper/models/ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e/base.pt",
    "small.en": "https://openaipublic.azureedge.net/main/whisper/models/f953ad0fd29cacd07d5a9eda5624af0f6bcf2258be67c92b79389873d91e0872/small.en.pt",
    "small": "https://openaipublic.azureedge.net/main/whisper/models/9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794/small.pt",
    "medium.en": "https://openaipublic.azureedge.net/main/whisper/models/d7440d1dc186f76616474e0ff0b3b6b879abc9d1a4926b7adfa41db2d497ab4f/medium.en.pt",
    "medium": "https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt",
    "large": "https://openaipublic.azureedge.net/main/whisper/models/e4b87e7e0bf463eb8e6956e646f1e277e901512310def2c24bf0e11bd3c28e9a/large.pt",

Currently whisper models are saved to by default(can download automatically):
C:\Users\"Username"\.cache\whisper

Recommended whisper model:
Use tiny for CPU 
tiny or small for GPU

-Silence threshold = Set higher than the microphone background noise for whisper stt
-Min Silent Chunks = Minimum amount of silent audio chunks in a row before the collected audio is transcribed. Also for whisper stt
-Stream ON = This allows the text to be played back more instantaneously when using the basic text to speech mode
-Chat History = Use chat history with Lm Studio. Turning on and off resets history
-Direct communication to applio api
-Applio autotune setting for singing(doesn't seem to work much)
-Whisper language selection. I included some popular ones but there are many more!
        en-English, es-Spanish, fr-French, de-German, it-Italian, ja-Japanese, ko-Korean, zh-CN-Chinese, ar-Arabic, hi-Hindi, ru-Russian, pt-Portuguese, pl-Polish, tr-Turkish, nl-Dutch, cs-Czech, hu-Hungarian
-Better and more user friendly GUI
-Added timeout adjuster for slower pc's
-Text field at the bottom so you can paste text you want to playback
-You can speak while the program is talking instead of having to wait for it to finish
-Added more speech filtering for misinterpreted commands

-Run setup.bat first. The pytorch cuda modules are big! might take up 5-6GB if you want to use whisper with GPU. You can always change to the non-cuda pytorch, torchvision, and torchaudio in the requirements.txt. Just remove the links to do so.

-Make sure LMstudio inference server is running
(url="http://localhost:1234/v1", api_key="lm-studio) Should be default setting
-Applio server also if you are using
http://127.0.0.1:6969/ Should be default setting

-You can change the system_message.txt to change how LM studio will respond to you.

max_tokens: The maximum number of [tokens](/tokenizer) that can be generated in the chat
completion.

temperature: What sampling temperature to use, between 0 and 2. Higher values like 0.8 will
make the output more random, while lower values like 0.2 will make it more
focused and deterministic.

frequency_penalty: Number between -2.0 and 2.0. Positive values penalize new tokens based on their
existing frequency in the text so far, decreasing the model's likelihood to
repeat the same line verbatim.

presence_penalty: Number between -2.0 and 2.0. Positive values penalize new tokens based on
whether they appear in the text so far, increasing the model's likelihood to
talk about new topics.

Commands

'no', 'nope', 'nah', 'negative', 'oh', 'note', 'now', 'know'
confirmation
'yes', 'ya', 'yep', 'yah', 'yet', 'yeah', 'affirmative', 'correct', 'yes sir', 'right', 'this', 'ok', 'okay', 'sure'
confirmation

'listen wake word', 'listen', 'wakeword', 'wake word', 'wait', 'listen wakeword', 'sleep'
Go back to listening for keyword

'exit', 'bye', 'end'
Exit the program

'shutdown system', 'shutdown the system', 'system shutdown', 'shutdown computer', 'shutdown the computer', 'computer shutdown', 'shut down system', 'shut down the system', 'system shut down', 'shut down computer', 'shut down the computer', 'computer shut down'
shutdown your computer

'clock', 'time', 'date', 'date time', 'time and date', 'date and time', 'time date', 'current time', 'current date', 'what is the time', 'what is the date', "today's date"
get time and date

'take note', 'save note', 'take notes', 'save notes', 'some notes', 'record some notes', 'save some notes', 'write some notes', "captain's log", 'captains log', 'star date', 'record log', 'write log', 'append log', 'save log'
save a note to text file

'open program', 'open a program', 'open an program', 'start program', 'start a program', 'start an program', 'launch program', 'launch a program', 'launch an program', 'open the program', 'start the program', 'launch the program', 'open application', 'open a application', 'open an application', 'launch application', 'launch a application', 'launch an application', 'start application', 'start a application', 'start an application', 'open the application', 'start the application', 'launch the application'
open a program

'close program', 'stop program', 'exit program', 'close the program', 'stop the prorgram', 'exit the program', 'close application', 'stop application', 'exit application', 'close a program', 'stop a program', 'exit a program', 'close a application', 'stop a application', 'exit a application', 'close an program', 'stop an program', 'exit an program', 'close an application', 'stop an application', 'exit an application','exit the application','close the application', 'stop the application'
close a program

'stop', 'stopped', 'stops', 'stuff' 'shut up', 'silence', 'quiet', 'top', 'dot' *sometimes stop is misinterperted as top or dot or the so this helps things function better*
interrupt a response(durring this process it checks all words in the string so regardless of what you say if it sees a stop command it will stop)