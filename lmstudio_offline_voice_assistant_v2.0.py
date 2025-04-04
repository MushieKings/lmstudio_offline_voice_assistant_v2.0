print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXxx++x$XXX$XXXxxxxX$$$8$$XxxxXXXXXXX$Xx+xxXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXx+::xX$$$$XXxxxXX$$$$888$$$XXXxxXX$$$$X+::xXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX+..:;+xX$$$$$$XXX$888888888888$XXX$$$$$$Xx+;:.:xXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX;..:+xxXX$$$$$$$$$$$$$$88888$$$$$$$$$$$$$XXXx+;..;XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX+++::;xXXXX$$$$$$$$$$$$$888$888$$$$$$$$$$$$XX$Xx;:;++xXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXx;::;;++xXXX$$$$$$$$$$$$88$$$$$88$$$$$$$$$$$XXXx++;;:;+xXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXx;...;+XXXXXXXX$XXXX$$$XX$$$$$$$$XX$$$XXX$$XXXXXXXX+;:.:;xXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXx+:.:;xXXx++xXXXXXXXXX;            :XXXXXX$XXx++xXXx;::;+XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXX+++++++xXXx+xX$Xx+.      .:+;;;..      .+xX$XxxxXXx+++++++XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXX+;..:+xXXx++xX;    :.....;;;;;;;:.....:    ;XxxxXXXx+:.:;+XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXX;:.:;+XXXx++XX:  .:;;;;;;;;;;;;;;;;;;;;:.  :XXx+xXXX+;::;+XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXx+;;;;+xxxx+     ....;;;:........:;;;:...    .xxxxx+;;;;+xXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXX+++::;xXXx+;   ......;:............:;......   ;xX$Xx;:;+++XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXx;:.:+xX$Xx+   .....:;:............:;:.....  .+xX$Xx+:.:;XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX+:..:;+xxXx   ::::;;;;:..........:;;;;::::  .xXxx+;:.:;xXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXx+++;;xXX+  .;;;.                    .;;;. .xXXx;;+++XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX;..;xXXx        .XxX; .Xxx+. :XXX.       .XXXx;..;XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXx;..:;++XX.   .xX$XX+ :$XXx. ;XX$Xx.   :XXx+;:..+xXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXx+;:+xX$X:  :;+xxXXXUwUXXXXxxx;:  :X$Xx+;;+xXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXxx;;;+++;.   ::..:::::::...::   .;+++;;+xXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")                                                                              
print("----------------------------------------LMstudio Offline Voice Assistant-v2.0------------------------------------------")
print("-----------------------------------------------Created by MushieKings--------------------------------------------------")
import customtkinter # GUI
import pyttsx3 #Bassic TTS
import pyaudio #play and record audio
import winsound #notification
import random #randomize the ai voice response
import time #For time.sleeps
import threading #Multithreading
import sys as sys #Used to exit or shut down computer
import os #Used to check file size or if file is there or write file
import tempfile #Delete applio temp files
import shutil #Delete applio temp files
import soundfile as sf # Get file duration
import numpy as np  
import whisper
import subprocess
from pygame import mixer #Play sounds and check file size
from AppOpener import open as appstart, close as appclose
#import lmstudio as lms
from openai import OpenAI #Interact with LmStudio
from vosk import Model, KaldiRecognizer #Speech to text
from datetime import datetime #Get date and time
import torch
print("Current working dir: ", os.getcwd()) #Current working Dir
print("torch version: ", torch.__version__)  # Check version  
print("CUDA = ", (torch.cuda.is_available()))  # Should print True if CUDA is available
print("If cuda False. Try uninstalling and reinstall the correct pytorch with cuda. https://pytorch.org/get-started/locally/")
torch.cuda.empty_cache()

# Instantiate configuration
client = OpenAI(base_url="http://127.0.0.1:1234", api_key="lm-studio")

# Basictts init defaults
basictts = pyttsx3.init('sapi5')
voices = basictts.getProperty('voices')
basictts.setProperty('voice',voices[0].id)
basictts.setProperty('rate', 223)
basictts.setProperty('volume', 1.0)

#LMSTUDIO System Settings-Read system message from file
def read_file_content(file_path):
    try:
        with open(file_path, "r") as file:
            return file.read().strip()
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None

system_message = read_file_content("system_message.txt")
if system_message is None:
    print("System message file not found, Defaulting to: You are the assistant and you're name is alfred.")
    system_message = "You are the assistant and you're name is alfred"
    basictts.say("System message file not found. Loading default")
    basictts.runAndWait()

#pygame init for notification sound
mixer.init()
try:
    up_sound = mixer.Sound('up.wav')
    down_sound = mixer.Sound('down.wav')
except:
    print('Failed to initialize up.wav and down.wav sound files')
    basictts.say("Failed to initialize sound files")
    basictts.runAndWait()

systemdelay = 1
voicestop = 0
###################################___SETTINGS___######################################
def get_settings():
    global voskmodel, hold_input, audio2text, ttsfilesize, rvcfilesize, ttsfile_path, rvcfile_path, conf_text, input_text, stop_text, applio_audio_file, voicestop
    voicestop = 0
    stop_text = ""
    conf_text = ""
    input_text = ""
    audio2text = ""
    print("Initializing Settings...")
    basictts.say("Initializing")
    basictts.runAndWait()
    #Get settings
    app.tts_radiovar = app.tts_radio_var.get() #Text to speech setting
    app.applio_folder = app.entry_applio_folder.get()   # GET APPLIO FOLDER
    app.applio_folder = app.applio_folder.replace( "\\", "/") 
    print("Applio Folder", app.applio_folder)
    ttsfile_path = (app.applio_folder + '/assets/audios/tts_output.wav')
    print("TTS Path: ", ttsfile_path)
    rvcfile_path = (app.applio_folder + '/assets/audios/tts_rvc_output.wav')
    print("RVC Path: ", rvcfile_path)
    # Check then create blank files if missing. Needed for detection on new file created
    if app.tts_radiovar == 1: 
        try:
            print("TTS file path: ", ttsfile_path)
            print("RVC file path: ", rvcfile_path)
            ttsfilesize = os.path.getsize(ttsfile_path)#has to get file size sooner or else error...
            rvcfilesize = os.path.getsize(rvcfile_path)
        except:
            print("one or more tts files not found or unreachable Applio\\assets\\audios (tts_output.wav) or (tts_rvc_output.wav)")
            open(ttsfile_path, 'w').close()
            open(rvcfile_path, 'w').close()
            ttsfilesize = os.path.getsize(ttsfile_path)#has to get file size sooner or else error...
            rvcfilesize = os.path.getsize(rvcfile_path)
            print("Created new blank files")
            time.sleep(0.1)
            pass
    app.stt_radiovar = app.stt_radio_var.get() #Speech to text setting
    #---------------------------FIRST COLUMN SETTINGS--------------------------BASIC TTS

    #---------------------------SECOND COLUMN SETTINGS-------------------------MAIN/LMSTUDIO
    #LMstudio settings
    try:
        app.lm_maxtokens = app.entrymt.get()
        app.lm_maxtokens = int(app.lm_maxtokens)
    except:
        basictts.say("Error loading max tokens")
        basictts.runAndWait()
        app.button_stop()
        return
    #GUI
    app.keyword = app.entrykw.get() #keyword setting
    app.keyword = app.keyword.lower() #keyword setting
    app.username = app.entryun.get() #get username
    #---------------------------THIRD COLUMN SETTINGS--------------------------APPLIO
    app.applio_voice_model = app.entry_applio_vm.get() # GET PTH FILE
    app.applio_index_file = app.entry_applio_if.get()   # GET INDEX FILE
    if app.tts_radiovar == 1 and app.ttsrvc == "RVC" and app.applio_voice_model == "" and app.applio_index_file == "": # Check not empty
        print("Applio voice model and or Index file input field empty")
        basictts.say("Applio voice model or Index file input field empty")
        basictts.runAndWait()
        app.button_stop()
        return
    applio_tts_voice = app.entry_applio_ttsv.get() # GET TTS VOICE
    if app.tts_radiovar == 1 and applio_tts_voice == "": # Check not empty
        print("Applio tts input field empty")
        basictts.say("Applio tts input field empty")
        basictts.runAndWait()
        app.button_stop()
        return
    if app.tts_radiovar == 1 and applio_tts_voice == "": # Check not empty
        print("Applio tts input field empty")
        basictts.say("Applio tts input field empty")
        basictts.runAndWait()
        app.button_stop()
        return
    #---------------------------FORTH COLUMN SETTINGS--------------------------VOSK/WHISPER
    #Vosk voice recognition init
    if app.stt_radiovar == 0:
        try:
            voskmodel=Model(app.entryvosk.get())
        except ValueError as ve:
            print(ve, "Invalid vosk model")
            app.button_stop()
    app.whispermodelname = app.entry_whisper_model.get() # get model name
    print("whispermodel:", app.whispermodelname)
    if app.whispermodelname.lower() in ["tiny", "small", "base", "medium", "large"]:    
        print("valid whisper model selected")
    else:
        print("Invalid whisper model name")
        basictts.say("Invalid whisper model name")
        basictts.runAndWait()
        app.button_stop()
        return
    try:
        app.silence_threshold = app.entry_st.get() # get silence threshold
        app.silence_threshold = float(app.silence_threshold)
    except ValueError as ve: 
        print(ve)
        print("Invalid silence threshold")
        basictts.say("Invalid silence threshold")
        basictts.runAndWait()
        app.button_stop()
        return
    try:
        app.minsilentchunks = app.entry_silent_chunks.get() # Amount of silent chunks before triggering transcription to text
        app.minsilentchunks = int(float(app.minsilentchunks))
    except ValueError as ve: 
        print(ve)
        print("Invalid silence threshold")
        basictts.say("Invalid silence threshold")
        basictts.runAndWait()
        app.button_stop()
        return
    hold_input = False
    app.autotune = app.autotune_switch.get()
    winsound.Beep(400, 100)
    return

#################################___VOICE INPUT___#######################################
#VOSK LISTEN THREAD
def vosk_listen():
    global hold_input, audio2text
    print("------------------------LISTENING WITH VOSK--------------------------")
    mic = pyaudio.PyAudio()
    recognizer = KaldiRecognizer(voskmodel, 16000)
    stream = mic.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=4096)
    stream.start_stream()
    counter = 0
    while app.startstop == "ON":
        counter += 1
        if counter >= 30:
            counter = 0
            print("Vosk listening...")
        try:
            data = stream.read(4096) #4096 or 8192 2048
        except OSError as e:
            print(f"Error reading audio data: {e}")
            continue
        if recognizer.AcceptWaveform(data):
            audio2text = recognizer.Result()
            audio2text = audio2text[14:-3]
            print("Transcribed Text: ", audio2text)
            #If keyword is alfred. Check if close enough and change to alfred.
            if app.keyword.lower() == "alfred" and audio2text.lower() in ['out there', "i'll fit", 'albert', 'ow, fred', 'alfredo', 'oh friends', 'health', 'else red', 'hail friend', "alfred's", 'helfrich', 'how fetched', 'of fridge', 'how fred', 'how it', 'how for it', 'how fair', 'how forward', 'how ford']: #misheard alfred
                print("misheard alfred, correcting")
                audio2text = "alfred"
                hold_input = True
            if audio2text.lower() in ['we cleared', 'like words', 'wait wait', 'just wait for it', 'required', 'dwayne cleared', 'why quit', 'when queried', 'wake where', 'why a quid', 'wait for it', 'wake weird']: #misheard alfred
                print("misheard wake word, correcting")
                audio2text = "wake word"
                hold_input = True
                
            if audio2text.lower() in ['you', 'thanks for watching', 'thank you', 'huh', 'harrison', 'scripts', 'ha', 'soups', 'about', 'each', 'ps', 'hook', 'hush', 'teach', 'stitched', 'this', 'huge', 'twitch', "it's such", 'she says', 'shh shh', 'shh shh shh', 'she', 'search', 'sheesh', 'where', 'whoa', 'oh', 'of', 'it', 'to', 'hersh', "i've", 'uh', 'he', 'health', 'she touched', 'shh', 'hatched', 'he sucks', 'barking', "who'd", 'she says', "it's", 'so', 'huh well', 'well', 'spring', 'usps', 'hopes', 'we', 'all', 'which', 'while', 'his', 'him', 'match', 'the', 'here', 'there', "he's", 'here', 'and', 'hips', 'just', "who's", 'that', 'new', 'her', 'in', 'then', 'too', 'next', 'last', 'more', 'first', 'long']: #vosk misinterpret noise[single words only]
                print('noise words detected, deleting')
                audio2text = ""
            if audio2text == None: #if None detected return ""
                print("audio2text None detected changing to '""'")
                audio2text = ""

            if audio2text != "": #if text is detected return text
                hold_input = True

            while hold_input == True and app.startstop == "ON":
                print("Holding input: ", audio2text)
                time.sleep(0.5)
                if hold_input == False:
                    print("Releasing hold of input")
                    break
                if audio2text == "":
                    break
                if app.startstop == "OFF":
                    break

        if app.startstop == "OFF":
            print("Stop detected in listen loop, Closing stream")
    else:
        print("Vosk listen loop stopping")
        return "shutdown"

#WHISPER LISTEN THREAD
def whisper_listen():
    global hold_input, audio2text
    print("------------------------LISTENING WITH WHISPER SUBPROCESS--------------------------")
    # Specify parameters
    device = app.whispercudacpu  #cuda or "cpu"
    print(device)
    app.whispermodelname
    print("Loading whisper model: ", app.whispermodelname)
    app.minsilentchunks
    silent_chunks = 0
    try:
        if torch.cuda.is_available():
            device = device
        else:
            print("Cuda unavailable. Using cpu.")
            device = "cpu"
        print("Device: ", device)
        whispermodel = whisper.load_model(app.whispermodelname, in_memory=True).to(device)  # Move the model to GPU
        whispermodel.eval() # Improve performance
        print("Model is on:", next(whispermodel.parameters()).device)  # Should print 'cuda:0' if on GPU
    except ValueError as ve:
        print(ve)

    def transcribe(audio_tensor):
        print("Processing audio...")
        with torch.no_grad():
            result = whispermodel.transcribe(audio_tensor, language=app.whisperlang) #---------------------OUTPUT----------------------- set language here possibly decode_options["language"] also "no_speech_threshold"  
        return result

    try:
        # Assume audio_chunk is your NumPy array  
        audio_chunk = np.random.rand(2048)  # Replace this with your actual audio data
        # Convert to PyTorch tensor  
        audio_tensor = torch.from_numpy(audio_chunk).float()  # Ensure it's float
        audio_tensor = audio_tensor.to(device) # Make sure your audio tensor is on the GPU
        # Initialize PyAudio  
        p = pyaudio.PyAudio()
        # Open a stream using float32 format  
        stream = p.open(format=pyaudio.paFloat32,
                        channels=1,
                        rate=16000,
                        input=True,
                        frames_per_buffer=2048) #8192 4096 2048 keep these numbers equal or you might get buffer overflow
        frames = []
    except ValueError as ve:
        print("Error:", ve, "ini np, pyaudio, torch")
    
    counter = 0
    while app.startstop == "ON":
        counter += 1
        if counter >= 20:
            counter = 0
            print("Whisper listening...")
        try:
            # Read audio data from the stream  
            data = stream.read(2048) #8192 4096 2048 keep these numbers equal or you might get buffer overflow
            audio_chunk = np.frombuffer(data, dtype=np.float32)
            audio_mean = np.abs(audio_chunk).mean()
            app.label_chunk_vol.configure(text=audio_mean)
        except OSError as e:
            print(f"Error reading audio data: {e}")
            
        # Check if the audio level exceeds the silence threshold  
        if audio_mean > app.silence_threshold:
            print("Audio detected.")
            frames.append(data)
            silent_chunks = 0  # Reset silent chunks if sound is detected  
        else:
            silent_chunks += 1  # Increment silent chunk count
        # If silence is detected for 'min_silent_chunks' chunks, process the recorded audio  
        if silent_chunks >= app.minsilentchunks:
            if frames:  # Only process if there are frames recorded  
                audio_data = b''.join(frames)  # Concatenate collected frames  
                frames = []  # Clear the list for the next batch
                # Convert byte data to NumPy array  
                audio_tensor = np.frombuffer(audio_data, dtype=np.float32).copy()
                # Perform speech recognition directly on the audio tensor  
                result = transcribe(audio_tensor)
                # Print the recognized text if it's not empty  
                if result['text'].strip():
                    result = result['text'].strip()
                    audio2text = result.lower().replace('.', '').replace('?', '') 
                    print("Transcribed Text: ", audio2text)
                    #If keyword is alfred. Check if close enough and change to alfred.
                    if app.keyword.lower() == "alfred" and audio2text.lower() in ['out there', "i'll fit", 'albert', 'ow, fred', 'alfredo', 'oh friends', 'health', 'else red', 'hail friend', "alfred's", 'helfrich', 'how fetched', 'of fridge', 'how fred', 'how it', 'how for it', 'how fair', 'how forward', 'how ford']: #misheard alfred
                        print("misheard alfred, correcting")
                        audio2text = "alfred"
                        hold_input = True
                    if audio2text.lower() in ['we cleared', 'like words', 'wait wait', 'just wait for it', 'required', 'dwayne cleared', 'why quit', 'when queried', 'wake where', 'why a quid', 'wait for it', 'wake weird']: #misheard alfred
                        print("misheard wake word, correcting")
                        audio2text = "wake word"
                        hold_input = True
                                        
                    if audio2text.lower() in ['you', 'thanks for watching', 'thank you', 'huh', 'harrison', 'scripts', 'ha', 'soups', 'about', 'each', 'ps', 'hook', 'hush', 'teach', 'stitched', 'this', 'huge', 'twitch', "it's such", 'she says', 'shh shh', 'shh shh shh', 'she', 'search', 'sheesh', 'where', 'whoa', 'oh', 'of', 'it', 'to', 'hersh', "i've", 'uh', 'he', 'health', 'she touched', 'shh', 'hatched', 'he sucks', 'barking', "who'd", 'she says', "it's", 'so', 'huh well', 'well', 'spring', 'usps', 'hopes', 'we', 'all', 'which', 'while', 'his', 'him', 'match', 'the', 'here', 'there', "he's", 'here', 'and', 'hips', 'just', "who's", 'that', 'new', 'her', 'in', 'then', 'too', 'next', 'last', 'more', 'first', 'long']: #vosk misinterpret noise[single words only]
                        print('noise words detected, deleting')
                        audio2text = ""
                    if audio2text == None: #if None detected return ""
                        print("audio2text None detected changing to '""'")
                        audio2text = ""
                    if audio2text != "": #if text is detected return text
                        hold_input = True
                    while hold_input == True and app.startstop == "ON":
                        print("Holding input: ", audio2text)
                        time.sleep(1)
                        if hold_input == False:
                            print("Releasing hold of input")
                            break
                        if audio2text == "":
                            break
                        if app.startstop == "OFF":
                            break
                if app.startstop == "OFF":
                    print("Stop detected in vosk listen loop, Closing stream")
                else:
                    print("")
            silent_chunks = 0  # Reset silence counter to start a new listening session
        time.sleep(0.005)  # Adjust this delay as needed
        if app.startstop == "OFF":
            try:
                torch.cuda.empty_cache()
                stream.stop_stream()
                stream.close()
                p.terminate()
            except ValueError as ve:
                print(ve)
                pass
            print("Whisper listen stopping")
            return "shutdown"


##################################___CHECK_INPUT___######################################
def wakeword():
    global hold_input, audio2text
    wake_text = ""
    counter = 0
    print("Init listening for wakeword...")
    while app.startstop == "ON":
        time.sleep(0.1)
        counter += 1
        if counter >= 100:
            counter = 0
            print("Listening for wakeword...")
        if hold_input == True and app.startstop == "ON":
            wake_text = audio2text
        if wake_text == app.keyword:
            hold_input = False
            print("Keyword Detected")
            basictts.say(("hi, " + app.username))
            basictts.runAndWait()
            return "input"
        if wake_text == "":
            hold_input = False
        if wake_text != "":
            hold_input = False
        if wake_text == None:
            wake_text = ""
            audio2text = ""
            hold_input = False
    else:
        print("Stopping wakeword")
        return "shutdown"

def input():
    global hold_input, voicestop, input_text
    input_text = ""
    mixer.stop()
    mixer.Sound.play(up_sound)
    mixer.stop()
    winsound.Beep(900, 100)
    counter = 0
    counter2 = 0
    while app.startstop == "ON":
        counter += 1
        if counter >= 20:
            counter = 0  
            print("Listening for input...")
        if hold_input == True and app.startstop == "ON":
            input_text = audio2text
            print("Input_text: ", input_text)
        if input_text == None:
            input_text = ""
            hold_input = False
        if input_text != "":
            winsound.Beep(400, 100)
            mixer.Sound.play(down_sound)
            mixer.stop()
            hold_input = False
            if app.confirm_on_off == "Confirm ON":
                return "confirm_input"
            else:
                return "input_check"
        time.sleep(0.1)
        counter2 += 1
        if counter2 >= 100 and app.startstop == "ON":
            counter2 = 0
            basictts.say("i'm listening")
            basictts.runAndWait()
    else:
        print("Stopping Input")
        return "shutdown"

def confirm_input():
    global hold_input, input_text, audio2text, conf_text
    print("confirming input")
    conf_text = ""
    audio2text = ""
    hold_input = False
    if app.startstop == "ON":
        didusay = (("did you say ", input_text))
        basictts.say(didusay)
        basictts.runAndWait()
        print("Confirm input Text: ", input_text)
    while app.startstop == "ON":
        if hold_input == True and app.startstop == "ON":
            print("Input_text: ", input_text)
            conf_text = audio2text
            print("Confirm_text: ", conf_text)
        if conf_text == None:
            conf_text = ""
        if conf_text != "":
            conf_list = conf_text.split(" ")
            for s in conf_list:
                words_in_s = set(s.lower().split())  # Split into distinct words
                if words_in_s.intersection({'no', 'nope', 'nah', 'negative'}):
                    basictts.say("ok, say again")
                    basictts.runAndWait()
                    hold_input = False
                    conf_text = ""
                    return "input"
                elif words_in_s.intersection({'yes', 'ya', 'yep', 'yah', 'yeah', 'affirmative', 'correct', 'right', 'ok', 'okay', 'sure', 'yup'}):
                    mixer.Sound.play(down_sound)
                    mixer.stop()
                    print("confirmed")
                    hold_input = False
                    conf_text = ""
                    return "input_check"
                else:
                    hold_input = False
                    rand1 = random.randrange(1, 1000)
                    if rand1 <= 500:
                        basictts.say("Sorry, didn't catch that")
                        basictts.runAndWait()
                    conf_text = ""
        if app.startstop == "ON":
            time.sleep(0.5)
            rand1 = random.randrange(1, 2000)
            if rand1 <=40:
                basictts.say(("awaiting confirmation, " + app.username))
                basictts.runAndWait()
            elif rand1 >=1960:
                basictts.say(("yes or no, " + app.username))
                basictts.runAndWait()
            rand2 = random.randrange(1, 2000)
            if rand2 <= 27:
                basictts.say("did you say ")
                basictts.runAndWait()
                print("Confirm Text: ", input_text)
                basictts.say(input_text)
                basictts.runAndWait()
    else:
        print("stop_button pressed in confirm_input")
        return "shutdown"

def input_check():
    global input_text, hold_input, voicestop
    hold_input = False
    voicestop = 0
    print("Checking Input...")
    if app.startstop == "ON":
        if input_text.lower() in ['listen wake word', 'listen', 'wakeword', 'wake word', 'wait', 'listen wakeword', 'sleep']:
            basictts.say(("ok i will listen for your command, ", app.username))
            basictts.runAndWait()
            return "listenwakeword"
        if input_text.lower() in ['exit', 'bye', 'end']:
            basictts.say(("Talk again soon, ", app.username))
            basictts.runAndWait()
            print("Exiting the conversation.")
            app.destroy()
            sys.exit()
        if input_text.lower() in ['shutdown system', 'shutdown the system', 'system shutdown', 'shutdown computer', 'shutdown the computer', 'computer shutdown', 'shut down system', 'shut down the system', 'system shut down', 'shut down computer', 'shut down the computer', 'computer shut down']:
            basictts.say("Initiating shutdown.")
            basictts.runAndWait()
            print("Shuting down computer.")
            os.system("shutdown /s /t 1")
            sys.exit()
        if input_text.lower() in ['time and day', 'clock', 'time', 'date', 'date time', 'time and date', 'date and time', 'time date', 'current time', 'current date', 'what is the time', 'what is the date', "today's date"]:
            return "timedate"
        if input_text.lower() in ['take note', 'save note', 'take notes', 'save notes', 'some notes', 'record some notes', 'save some notes', 'write some notes', "captain's log", 'captains log', 'star date', 'record log', 'write log', 'append log', 'save log']:
            basictts.say(("Ready to take your notes, ", app.username))
            basictts.runAndWait()
            return "takenotes"
        if input_text.lower() in ['open program', 'open a program', 'open an program', 'start program', 'start a program', 'start an program', 'launch program', 'launch a program', 'launch an program', 'open the program', 'start the program', 'launch the program', 'open application', 'open a application', 'open an application', 'launch application', 'launch a application', 'launch an application', 'start application', 'start a application', 'start an application', 'open the application', 'start the application', 'launch the application']:
            basictts.say("What program would you like to open?")
            basictts.runAndWait()
            return "openapp"
        if input_text.lower() in ['close program', 'stop program', 'exit program', 'close the program', 'stop the prorgram', 'exit the program', 'close application', 'stop application', 'exit application', 'close a program', 'stop a program', 'exit a program', 'close a application', 'stop a application', 'exit a application', 'close an program', 'stop an program', 'exit an program', 'close an application', 'stop an application', 'exit an application','exit the application','close the application', 'stop the application']:
            basictts.say("What program would you like to close?")
            basictts.runAndWait()
            return "closeapp"
        else:
            return "lmstudio_api" 
    else:
        return "shutdown"
####################################___DATA_PROCESSING___################################
def stoplistener(): #Listen for stop command during data processing phase
    global voicestop, hold_input, stop_text
    voicestop = 0
    hold_input = False
    counter = 0
    while app.startstop == "ON" and voicestop == 0:
        if voicestop == 1:
            break
        counter += 1
        if counter >= 10:
            counter = 0
            print("\n----------------Listening for stop command-----------------")
        time.sleep(0.5)
        if hold_input == True and voicestop == 0 and app.startstop == "ON":
            stop_text = audio2text
        if stop_text == None:
            stop_text = ""
            hold_input = False
        stop_list = stop_text.split(" ")
        for s in stop_list:
            if s.lower() in ['stop', 'stopped', 'stops', 'stuff', 'shut up', 'silence', 'quiet', 'top', 'dot']:
                print("voicestop detected, stopping")
                voicestop = 1
                try:
                    applio_audio_file.stop()
                except ValueError as ve:
                    print(ve)
                    continue
                print("Stop detected, exiting listener back to input")
                return "input"
        hold_input = False
    else:
        return "shutdown"

def lmstudio_api(): #Tell Lmstudio to generate output
    global hold_input, input_text, model_response, voicestop
    model_response = "GeneratingResponse"
    mixer.Sound.play(up_sound)
    mixer.stop()
    winsound.Beep(900, 100)
    hold_input = False
    #STREAMING?
    #-----------------------------Chat History OFF--------Stream ON BTTS only---------------------------------------------------
    words = ""
    try:
        if app.startstop == "ON" and app.chat_history == "OFF" and app.btts_stream_on == True and voicestop == 0:
            while app.startstop == "ON" and voicestop == 0:
                response = client.chat.completions.create(
                    model="local-model",
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": input_text}
                    ],
                    temperature=app.lm_temp,
                    max_tokens=app.lm_maxtokens,
                    frequency_penalty=app.lm_freq_p_val,
                    presence_penalty=app.lm_presence_p_val,
                    stream=True,
                )

                new_message = {"role": "assistant", "content": ""}

                for chunk in response:
                    if chunk.choices[0].delta.content:
                        print(chunk.choices[0].delta.content, end="", flush=True)
                        words += chunk.choices[0].delta.content
                        if any(punct in words for punct in ['.', '?', '!', ',']):
                            basictts.say(words)  # Speak the contents of the words variable  
                            basictts.runAndWait()  # Wait for the speech to finish  
                            words = ""  # Clear the words variable
                        new_message["content"] += chunk.choices[0].delta.content
                    if app.startstop == "ON" and voicestop == 1:
                        break
                    
                print("Model Response: ", new_message["content"])
                model_response = new_message["content"]
                return "input"
        #------------------------------Chat History ON---------Stream ON BTTS only-----------------------------------------------------------
        if app.startstop == "ON" and app.chat_history == "ON" and app.btts_stream_on == True and voicestop == 0:
            app.history.append({"role": "user", "content": input_text})
            while app.startstop == "ON" and voicestop == 0:
                response = client.chat.completions.create(
                    model="local-model",
                    messages=app.history,
                    temperature=app.lm_temp,
                    max_tokens=app.lm_maxtokens,
                    frequency_penalty=app.lm_freq_p_val,
                    presence_penalty=app.lm_presence_p_val,
                    stream=True,
                )

                new_message = {"role": "assistant", "content": ""}

                for chunk in response:
                    if chunk.choices[0].delta.content:
                        print(chunk.choices[0].delta.content, end="", flush=True)
                        words += chunk.choices[0].delta.content
                        if any(punct in words for punct in ['.', '?', '!', ',']):
                            basictts.say(words)  # Speak the contents of the words variable  
                            basictts.runAndWait()  # Wait for the speech to finish  
                            words = ""  # Clear the words variable
                        new_message["content"] += chunk.choices[0].delta.content
                    if app.startstop == "ON" and voicestop == 1:
                        break
                model_response = new_message["content"]
                app.history.append(new_message)
                return "input"
        #-----------------------------Chat History OFF--------Stream OFF---------------------------------------------------
        if app.startstop == "ON" and app.chat_history == "OFF" and app.btts_stream_on == False and voicestop == 0:
            response = client.chat.completions.create(
                model="local-model",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": input_text}
                ],
                temperature=app.lm_temp,
                max_tokens=app.lm_maxtokens,
                frequency_penalty=app.lm_freq_p_val,
                presence_penalty=app.lm_presence_p_val,
                stream=False
            )
            print("Model Response: ", response.choices[0].message.content.strip())
            model_response = response.choices[0].message.content.strip()
            print("Initiate conversation complete")
        #------------------------Chat History ON---------------Stream OFF-----------------------------------------------------------
        if app.startstop == "ON" and app.chat_history == "ON" and app.btts_stream_on == False and voicestop == 0:
            app.history.append({"role": "user", "content": input_text})
            response = client.chat.completions.create(
                    model="local-model",
                    messages=app.history,
                    temperature=app.lm_temp,
                    max_tokens=app.lm_maxtokens,
                    frequency_penalty=app.lm_freq_p_val,
                    presence_penalty=app.lm_presence_p_val,
                    stream=False
                )

            new_message = {"role": "assistant", "content": ""}
            #for chunk in response:
            #    if chunk.choices[0].delta.content:
            #        print(chunk.choices[0].delta.content, end="", flush=True)
            new_message["content"] = response.choices[0].message.content.strip()
            app.history.append(new_message)
            print("Model Response: ", new_message["content"])
            model_response = new_message["content"]
            print("Initiate conversation complete")
    except Exception as ce:
        print(ce)
        print("Failed to connect to lm studio")
        voicestop = 1
        basictts.say("Failed to connect to lm studio")
        basictts.runAndWait()
        return "input"

    if voicestop == 1:
        print("Voice command stopped")
        return "input"

    if app.startstop == "OFF":
        return "shutdown"

    if app.tts_radiovar == 0:
        print("Sending response to basic tts")
        return "basictts_response"
    if app.tts_radiovar == 1:
        print("Sending response to applio api")
        return "run_applio_worker"

def basictts_response(): #Playback with windows TTS
    global voicestop, hold_input
    hold_input = False
    replace_txt = model_response.replace('Mr.', 'Mr').replace('Mrs.', 'Mrs').replace('<|eot_id|>', '').replace('*', ' ').replace('\n', '.').replace('\t', ' ').replace('<|start_header_id|>', '').replace('<|end_header_id|>', '').replace('user', '') #.replace(',', '.')
    split_txt = replace_txt.split('.')
    if app.startstop == "ON" and voicestop == 0 and app.btts_stream_on == False and model_response !="":
        for x in split_txt:
            if app.startstop == "ON" and voicestop == 0:
                basictts.say(x)
                print(x)
                basictts.runAndWait()
            if voicestop == 1:
                basictts.stop()
                print("Stopping basic TTS response")
                return "input"
            if app.startstop == "OFF":
                break
        else:
            print("Ending basic TTS response")
            basictts.say("returning to input")
            basictts.runAndWait()
            voicestop = 1
            return "input"
    if app.startstop == "OFF":
        return "shutdown"

def run_applio_worker(): #Send text to applio api to generate audio
    global hold_input, ttsfilesize, rvcfilesize, ttsfile_path, rvcfile_path, ttsfile_mod_time, rvcfile_mod_time
    hold_input = False
    ttsfilesize = os.path.getsize(ttsfile_path)
    rvcfilesize = os.path.getsize(rvcfile_path)
    print("ttsfile file size: ", ttsfilesize)
    print("rvcfile file size: ", rvcfilesize)
    ttsfile_mod_time = os.path.getmtime(ttsfile_path) #last modified for save check
    rvcfile_mod_time = os.path.getmtime(rvcfile_path) #last modified for save check
    print("ttsfile Last modified: ", ttsfile_mod_time)
    print("rvcfile Last modified: ", rvcfile_mod_time)
    print("Convert text to audio...")
    print("TTS mode: ", app.ttsrvc)
    model_response_cleaned = model_response.replace('<|eot_id|>', '').replace('*', ' ').replace('\n', '. ').replace('\t', ' ').replace('<|start_header_id|>', '').replace('<|end_header_id|>', '').replace('user', '')
    # Prepare the command
    command = ['python', 'applio_worker.py',
        model_response_cleaned,
        app.applio_tts_voice,
        str(app.applio_rate),
        str(app.pitch),
        app.applio_folder,
        app.applio_voice_model,
        app.applio_index_file,
        app.ttsrvc,
        app.autotune
    ]

    # Start the subprocess
    process = subprocess.Popen(command)
    
    try:
        while True:
            time.sleep(2)  # Check every second
            hold_input = False
            # Check if the subprocess has finished
            if process.poll() is not None:
                print("Subprocess completed.")
                process.terminate()  # Send SIGTERM signal to terminate the process
                process.wait()  # Wait for the process to terminate completely
                break  # Exit the loop if the subprocess has completed

            # Check the voicestop variable
            if voicestop == 1:
                print("Stopping the subprocess...")
                process.terminate()  # Send SIGTERM signal to terminate the process
                process.wait()  # Wait for the process to terminate completely
                print("Returning to input")
                return "input"
            
            if app.startstop == "OFF":
                print("Stopping the subprocess...")
                process.terminate()  # Send SIGTERM signal to terminate the process
                process.wait()  # Wait for the process to terminate completely
                return "shutdown"

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Check the return code
        return_code = process.returncode
        if return_code is not None and return_code != 0:
            print("Error occurred while executing subprocess. Check applio console. Ignoring error.")
            process.terminate()  # Send SIGTERM signal to terminate the process
            process.wait()  # Wait for the process to terminate completely
            return "play_applio_audio"  # Or handle this case as needed

    process.terminate()  # Send SIGTERM signal to terminate the process
    process.wait()  # Wait for the process to terminate completely
    print("Returning to 'play_applio_audio'")
    return "play_applio_audio"

def play_applio_audio(): #Playback completed audio
    global voicestop, hold_input, applio_audio_file, ttsfilesize, rvcfilesize, ttsfile_path, rvcfile_path, ttsfile_mod_time, rvcfile_mod_time
    hold_input = False
    print("Starting save file check...")
    print("ttsfile Last modified: ", ttsfile_mod_time)
    print("rvcfile Last modified: ", rvcfile_mod_time)
    print("playback:", app.ttsrvc)
    #FLAGS
    applio_audio_file = mixer.Sound
    fileready = False
    savedone = False
    def wait_for_file_ready(file_path, file_mod_time): #WAIT FOR FILE TO SAVE
        filecounter = 0
        
        while voicestop == 0 and app.startstop == "ON" and filecounter < app.applio_timeout:
            current_mod_time = os.path.getmtime(file_path)
            if current_mod_time == file_mod_time:
                print("File save started.")
                time.sleep(1)
                return True
            
            file_mod_time = current_mod_time
            filecounter += 1
            time.sleep(1)
            print("New file not detected yet. Timeout in: ", filecounter, " of ", app.applio_timeout)
            
            if filecounter >= app.applio_timeout:
                app.button_stop()
                return False
            
            if app.startstop == "OFF":
                return "shutdown"

    def wait_for_file_save(file_path, file_size): #WAIT FOR FILE TO SAVE
        filecounter = 0
        
        while voicestop == 0 and app.startstop == "ON" and filecounter < app.applio_timeout:
            current_file_size = os.path.getsize(file_path)
            if current_file_size == file_size:
                print("File is same size, save complete")
                time.sleep(3)
                return True
            
            file_size = current_file_size
            filecounter += 1
            time.sleep(1)
            print("Still saving file. Timeout in: ", filecounter, " of ", app.applio_timeout)
            
            if filecounter >= app.applio_timeout:
                app.button_stop()
                return False
            
            if app.startstop == "OFF":
                return "shutdown"

    if app.ttsrvc == "TTS" and app.startstop == "ON":
        if fileready == False:
            fileready = wait_for_file_ready(ttsfile_path, ttsfile_mod_time)
        if savedone == False:
            savedone = wait_for_file_save(ttsfile_path, ttsfilesize)
    elif app.ttsrvc == "RVC" and app.startstop == "ON":
        if fileready == False:
            wait_for_file_ready(rvcfile_path, rvcfile_mod_time)
        if savedone == False:
            savedone = wait_for_file_save(rvcfile_path, rvcfilesize)
            
    tts_file_size = os.path.getsize(ttsfile_path)
    if tts_file_size < 100 and savedone == True and app.ttsrvc == "TTS":
        print("Error file size too small or never saved. Maybe error executing applio api")
        app.button_stop()
        return
    rvc_file_size = os.path.getsize(rvcfile_path)
    if rvc_file_size < 100 and savedone == True and app.ttsrvc == "RVC":
        print("Error file size too small or never saved. Maybe error executing applio api")
        app.button_stop()
        return

    if app.startstop == "OFF":
        return "shutdown"
    
    playaudiovar = "play"
    print("Loading audio to play...")

    def play_audio(tts_path): #PLAY AUDIO AND GET LENGTH
        global voicestop, hold_input
        hold_input = False
        try:
            data, sample_rate = sf.read(tts_path)
            # Calculate duration
            durtime = len(data) / sample_rate
            duration_seconds = durtime
            duration_seconds = int(duration_seconds)
            duration_seconds += 1
            print(f"Duration: {duration_seconds} seconds")
            applio_audio_file = mixer.Sound(tts_path)  # Initialize response playback
            applio_audio_file.play()
            elapsed_time = 0
            while app.startstop == "ON" and voicestop == 0:
                print("Playtime: ", elapsed_time + 1, "s of ", duration_seconds)  # Display as 1-based
                time.sleep(1)
                elapsed_time += 1

                # Stop playback if voicestop changes
                if voicestop == 1:
                    playaudiovar == "delete_applio_temp"
                    break
                
                # If the audio has finished playing
                if elapsed_time >= duration_seconds:
                    voicestop = 1
                    playaudiovar == "delete_applio_temp"
                    break

            print("Playback stopped")
            applio_audio_file.stop()
            voicestop = 1
            return "delete_applio_temp"
    
        except Exception as error:
            print(error)
            time.sleep(0.1)
            applio_audio_file.stop()
            playaudiovar == "delete_applio_temp"

    if app.startstop == "ON" and voicestop == 0 and playaudiovar == "play":
        if app.ttsrvc == "TTS":
            print("playback tts", app.ttsrvc)
            playaudiovar = play_audio(ttsfile_path)
        elif app.ttsrvc == "RVC":
            print("playback rvc:", app.ttsrvc)
            playaudiovar = play_audio(rvcfile_path)

    if app.startstop == "OFF":
        return "shutdown"

    if playaudiovar == "delete_applio_temp":
        voicestop = 1
        return "delete_applio_temp"

def delete_applio_temp(): #Delete excess audio files in gradio temp folder
    global hold_input, voicestop
    hold_input = False
    # Determine the location of Gradio's temp directory
    gradio_temp_dir = os.path.join(tempfile.gettempdir(), 'gradio')
    # Check if the directory exists and delete its contents
    if os.path.exists(gradio_temp_dir):
        try:
            shutil.rmtree(gradio_temp_dir)  # Deletes the directory and all its contents
            print(f"Cleared Gradio temp directory: {gradio_temp_dir}")
        except Exception as e:
            print(f"Failed to clear Gradio temp directory: {e}")
    else:
        print(f"Gradio temp directory does not exist: {gradio_temp_dir}")
        time.sleep(0.2)
        voicestop = 0
    if app.startstop == "OFF":
        return "shutdown"
    return "input"

#######################################___FEATURES___####################################
def time_and_date():
    global hold_input, input_text
    hold_input = False
    if app.startstop == "ON":
        timestamp = time.time()
        timestamp = int(timestamp)
        date_time = datetime.fromtimestamp(timestamp)
        timevar = date_time.strftime("The time is %I %M %p")
        datevar = date_time.strftime("Today is %A, %B %d, %Y")
        print(timevar)
        basictts.say(timevar)
        basictts.runAndWait()
        print(datevar)
        basictts.say(datevar)
        basictts.runAndWait()
        basictts.say("returning to input")
        basictts.runAndWait()
        return "input"
    
def take_note():
    global hold_input, input_text
    hold_input = False
    while app.startstop == "ON":
        if input() == "confirm_input":
            if confirm_input() == "input_check":
                break
            else:
                continue
        else:
            return "shutdown"
    filename = app.entrynote.get()
    timestamp = time.time()
    timestamp = int(timestamp)
    date_time = datetime.fromtimestamp(timestamp)
    timevar = date_time.strftime("The time is %I:%M %p")
    datevar = date_time.strftime("Today is %A, %B %d, %Y")
    txt_file = open((str(filename) + ".txt"), "a")
    txt_file.write((timevar + ", " + datevar + "\n" + "-" + input_text + "\n"))
    txt_file.close()
    print(input_text)
    print("Saved")
    basictts.say("Saved, returning to input")
    basictts.runAndWait()
    return "input"

def open_app():
    global hold_input, input_text
    hold_input = False
    while True and app.startstop == "ON":
        if input() == "confirm_input":
            if confirm_input() == "input_check":
                break
            else:
                continue
    try:
        appstart(input_text, match_closest=True)
    except:
        basictts.say("failed to open")
        basictts.runAndWait()
    basictts.say("returning to input")
    basictts.runAndWait()
    return "input"

def close_app():
    global hold_input, input_text
    hold_input = False
    while True and app.startstop == "ON":
        if input() == "confirm_input":
            if confirm_input() == "input_check":
                break
            else:
                continue
    try:
        appclose(input_text, match_closest=True, output=False)
    except:
        basictts.say("failed to close")
        basictts.runAndWait()
    basictts.say("returning to input")
    basictts.runAndWait()

########################################################################################################
########################################################################################################
########################################################################################################
def main_loop():
    get_settings()
    print("Settings loaded")

    if app.stt_radiovar == 0: #VOSK
        print("Init Vosk")
        vosk_listen_thread = threading.Thread(target=vosk_listen)
        vosk_listen_thread.start()
    if app.stt_radiovar == 1: #WHISPER
        print("Starting whisper thread")
        whisper_listen_thread = threading.Thread(target=whisper_listen)
        whisper_listen_thread.start()

    main = "listenwakeword"
    while True:
        #VOICE INPUT
        if main == "input":
            main = input()
        #CHECK INPUT
        if main == "listenwakeword":
            main = wakeword()
        if main == "confirm_input":
            main = confirm_input()
        if main == "input_check":
            main = input_check()
        #DATA PROCESSING
        if main == "lmstudio_api":
            stop_listener_thread = threading.Thread(target=stoplistener)
            stop_listener_thread.start()
            main = lmstudio_api()
        if main == "basictts_response":
            main = basictts_response()
        if main == "run_applio_worker":
            main = run_applio_worker()
        if main == "play_applio_audio":
            main = play_applio_audio()
        if main == "delete_applio_temp":
            main = delete_applio_temp()
        #FEATURES
        if main == "timedate":
            main = time_and_date()
        if main == "takenotes":
            main = take_note()
        if main == "openapp":
            main = open_app()
        if main == "closeapp":
            main = close_app()
        if main == "shutdown":
            print("Shutting Down main loop")
            break
########################################################################################################
########################################################################################################
########################################################################################################

########____GUI____##########
customtkinter.set_appearance_mode("dark")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"
class App(customtkinter.CTk):
###############___GUI___###############
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.geometry("690x660")
        self.title("LMstudio Offline Voice Assistant v2.0")
        self.frame = customtkinter.CTkFrame(master=self)
        self.frame.grid(padx=7, pady=10, row=20, column=3, rowspan=4, sticky="nsew")
        #GUI AND VARIABLE DEFAULTS
        self.tts_radio_var = customtkinter.IntVar(value=0)
        self.tts_radiovar = 0
        self.stt_radio_var = customtkinter.IntVar(value=0)
        self.stt_radiovar = 0
        #1st column BTTS------------------------------------------------------------
        #BASIC WINDOWS TTS
        self.btts_radio_button = customtkinter.CTkRadioButton(master=self.frame, variable=self.tts_radio_var, value=0, text="Basic TTS", font=("Impact", 14), text_color="gold")
        self.btts_radio_button.grid(row=1, column=1, pady=3, padx=6)
        #CHAT HISTORY ON OFF
        self.chat_history_button = customtkinter.CTkButton(master=self.frame, hover=False, fg_color="blue", text="Chat History OFF", font=("Impact", 18), text_color="white", command=self.button_chat_history_on)
        self.chat_history_button.grid(row=2, column=1, pady=3, padx=6)
        self.chat_history = "OFF" #Default setting
        self.history = [{"role": "system", "content": system_message}]
        #VOICEID
        self.labelbttsvoiceid = customtkinter.CTkLabel(master=self.frame, text="BTTS Voiceid: 0", font=("Impact", 18), text_color="deep sky blue")
        self.labelbttsvoiceid.grid(row=3, column=1, pady=3, padx=6)
        self.btts_voiceid_slider = customtkinter.CTkSlider(self.frame, width=150, from_=0, to=9, number_of_steps=10, command=self.btts_voiceid_slider_update)
        self.btts_voiceid_slider.grid(row=4, column=1, padx=3, pady=6)
        self.btts_voiceid_slider.set(0)
        #BTTS SPEED
        self.labelbttsrate = customtkinter.CTkLabel(master=self.frame, text="BTTS Rate: 223", font=("Impact", 18), text_color="deep sky blue")
        self.labelbttsrate.grid(row=5, column=1, pady=3, padx=6)
        self.btts_rate_slider = customtkinter.CTkSlider(self.frame, width=150, from_=1, to=400, number_of_steps=40, command=self.btts_rate_slider_update)
        self.btts_rate_slider.grid(row=6, column=1, padx=3, pady=6)
        self.btts_rate_slider.set(223)
        #BTTS VOLUME
        self.labelbttsvol = customtkinter.CTkLabel(master=self.frame, text="BTTS Vol: 1.0", font=("Impact", 18), text_color="deep sky blue")
        self.labelbttsvol.grid(row=7, column=1, pady=3, padx=6)
        self.bttsvol_slider = customtkinter.CTkSlider(self.frame, width=150, from_=0, to=1, number_of_steps=10, command=self.btts_volume_slider_update)
        self.bttsvol_slider.grid(row=8, column=1, padx=3, pady=6)
        self.bttsvol_slider.set(1.0)
        #FASTER BTTS STREAM
        self.labelstream = customtkinter.CTkLabel(master=self.frame, text="Faster BTTS", font=("Impact", 18), text_color="deep sky blue")
        self.labelstream.grid(row=9, column=1, pady=3, padx=6)
        self.stream_onoff_button = customtkinter.CTkButton(master=self.frame, hover=False, fg_color="blue", text="Stream OFF", font=("Impact", 18), text_color="white", command=self.button_stream_on)
        self.stream_onoff_button.grid(row=10, column=1, pady=3, padx=6)
        self.btts_stream_on = False
        #LOG FILE
        self.labelnote = customtkinter.CTkLabel(master=self.frame, text="Log file Name", font=("Impact", 18), text_color="deep sky blue")
        self.labelnote.grid(row=11, column=1, pady=3, padx=6)
        self.entrynote = customtkinter.CTkEntry(master=self.frame, placeholder_text="file name")
        self.entrynote.insert(0, "captains_log")
        self.entrynote.grid(row=12, column=1, pady=3, padx=6)
        #GUI SCALING
        self.labelscaling = customtkinter.CTkLabel(master=self.frame, text="GUI Scale", font=("Impact", 18), text_color="deep sky blue")
        self.labelscaling.grid(row=13, column=1, pady=3, padx=6)
        self.scaling_optionemenu = customtkinter.CTkOptionMenu(self.frame, values=["80%", "90%", "100%", "110%", "120%"], command=self.change_scaling_event)
        self.scaling_optionemenu.grid(row=14, column=1, padx=20, pady=(10, 20))
        self.scaling_optionemenu.set("100%")
        #CONFIRM TOGGLE BUTTON
        self.confirm_on_off_switch = customtkinter.CTkSwitch(master=self.frame, text="Confirm ON", font=("Impact", 16), text_color="green", onvalue="Confirm ON", offvalue="Confirm OFF", state="on", command=self.confirm_on_off_switch_update)
        self.confirm_on_off_switch.grid(row=15, column=1, padx=3, pady=6)
        self.confirm_on_off_switch.select()
        self.confirm_on_off = "Confirm ON"

        #2nd column LM studio---------------------------------------------------------------------------------
        #TITLE FOR RADIO BUTTONS
        self.label_tts = customtkinter.CTkLabel(master=self.frame, text="Text to Speech", font=("Impact", 20), text_color="gold")
        self.label_tts.grid(row=1, column=2, pady=3, padx=6)
        #START BUTTON
        self.start_button = customtkinter.CTkButton(master=self.frame, hover=False, fg_color="green", text="Start", font=("Impact", 18), text_color="white", command=self.button_start)
        self.start_button.grid(row=2, column=2, pady=3, padx=6)
        #KEYWORD
        self.labelkw = customtkinter.CTkLabel(master=self.frame, text="Keyword", font=("Impact", 18), text_color="deep sky blue")
        self.labelkw.grid(row=3, column=2, pady=3, padx=6)
        self.entrykw = customtkinter.CTkEntry(master=self.frame, placeholder_text="alfred")
        self.entrykw.insert(0, "alfred")
        self.entrykw.grid(row=4, column=2, pady=3, padx=6)
        self.keyword = "alfred"
        #USER NAME
        self.labelun = customtkinter.CTkLabel(master=self.frame, text="User Name", font=("Impact", 18), text_color="deep sky blue")
        self.labelun.grid(row=5, column=2, pady=3, padx=6)
        self.entryun = customtkinter.CTkEntry(master=self.frame, placeholder_text="Boss")
        self.entryun.insert(0, "Boss")
        self.entryun.grid(row=6, column=2, pady=3, padx=6)
        self.username = "Boss"
        #MAX TOKENS
        self.labelmt = customtkinter.CTkLabel(master=self.frame, text="Max Tokens", font=("Impact", 18), text_color="light blue")
        self.labelmt.grid(row=7, column=2, pady=3, padx=6)
        self.entrymt = customtkinter.CTkEntry(master=self.frame, placeholder_text="-1")
        self.entrymt.insert(0, -1)
        self.entrymt.grid(row=8, column=2, pady=3, padx=6)
        self.lm_maxtokens = -1
        #TEMPERATURE
        self.labeltemp = customtkinter.CTkLabel(master=self.frame, text="Temperature: 0.7", font=("Impact", 18), text_color="light blue")
        self.labeltemp.grid(row=9, column=2, pady=3, padx=6)
        self.tempurature_slider = customtkinter.CTkSlider(self.frame, width=150, from_=-2.0, to=2.0, number_of_steps=40, command=self.lm_tempurature_slider_update)
        self.tempurature_slider.grid(row=10, column=2, padx=3, pady=6)
        self.tempurature_slider.set(0.7)
        self.lm_temp = 0.7
        #FREQUENCY PENALTY
        self.labelfp = customtkinter.CTkLabel(master=self.frame, text="FreqPenalty: 0.0", font=("Impact", 18), text_color="light blue")
        self.labelfp.grid(row=11, column=2, pady=3, padx=6)
        self.freqpen_slider = customtkinter.CTkSlider(self.frame, width=150, from_=-2.0, to=2.0, number_of_steps=40, command=self.lm_frequency_penalty_slider_update)
        self.freqpen_slider.grid(row=12, column=2, padx=3, pady=6)
        self.freqpen_slider.set(0.0)
        self.lm_freq_p_val = 0.0
        #PRESENCE PENALTY
        self.labelpp = customtkinter.CTkLabel(master=self.frame, text="PresPenalty: 0.0", font=("Impact", 18), text_color="light blue")
        self.labelpp.grid(row=13, column=2, pady=3, padx=6)
        self.prespen_slider = customtkinter.CTkSlider(self.frame, width=150, from_=-2.0, to=2.0, number_of_steps=40, command=self.lm_presence_penalty_slider_update)
        self.prespen_slider.grid(row=14, column=2, padx=3, pady=6)
        self.prespen_slider.set(0.0)
        self.lm_presence_p_val = 0.0
        #APPLIO SAVE FILE TIMEOUT LABEL
        self.labelappliotimeout = customtkinter.CTkLabel(master=self.frame, text="Applio Save Timeout: 15", font=("Impact", 14), text_color="gold")
        self.labelappliotimeout.grid(row=15, column=2, columnspan=1, pady=3, padx=6)

        #3rd column applio--------------------------------------------------------
        #APPLIO MODE RADIO BUTTON
        self.applio_radio_button = customtkinter.CTkRadioButton(master=self.frame, variable=self.tts_radio_var, value=1, text="Applio TTS", font=("Impact", 14), text_color="gold", command=self.applio_not_stream)
        self.applio_radio_button.grid(row=1, column=3, pady=3, padx=6)
        #APPLIO TTS OR RVC BUTTON
        self.ttsrvc_button = customtkinter.CTkButton(master=self.frame, hover=False, fg_color="deep sky blue", text="Applio TTS", font=("Impact", 18), text_color="white", command=self.play_rvc)
        self.ttsrvc_button.grid(row=2, column=3, pady=3, padx=6)
        self.ttsrvc = "TTS"
        #APPLIO FOLDER
        self.label_applio_folder = customtkinter.CTkLabel(master=self.frame, text="Applio Folder", font=("Impact", 18), text_color="deep sky blue")
        self.label_applio_folder.grid(row=3, column=3, pady=3, padx=6)
        self.entry_applio_folder = customtkinter.CTkEntry(master=self.frame, placeholder_text=r"C:\AI\Applio")
        self.entry_applio_folder.insert(0, "C:\\AI\\Applio")
        self.entry_applio_folder.grid(row=4, column=3, pady=3, padx=6)
        self.applio_folder = "C:\\AI\\Applio"
        #TTS VOICE
        self.label_applio_ttsv = customtkinter.CTkLabel(master=self.frame, text="TTS Voice", font=("Impact", 18), text_color="deep sky blue")
        self.label_applio_ttsv.grid(row=5, column=3, pady=3, padx=6)
        self.entry_applio_ttsv = customtkinter.CTkEntry(master=self.frame, placeholder_text="en-US-AndrewNeural")
        self.entry_applio_ttsv.insert(0, "en-US-AndrewNeural")
        self.entry_applio_ttsv.grid(row=6, column=3, pady=3, padx=6)
        self.applio_tts_voice = "en-US-AndrewNeural"
        #APPLIO TTS RATE
        self.label_applio_rate = customtkinter.CTkLabel(master=self.frame, text="TTS Rate: 0", font=("Impact", 18), text_color="deep sky blue")
        self.label_applio_rate.grid(row=7, column=3, pady=3, padx=6)
        self.applio_rate_slider = customtkinter.CTkSlider(self.frame, width=150, from_=-100, to=100, number_of_steps=40, command=self.applio_rate_slider_update)
        self.applio_rate_slider.grid(row=8, column=3, padx=3, pady=6)
        self.applio_rate_slider.set(0.0)
        self.applio_rate = 0.0
        #RVC VOICE MODEL
        self.label_applio_vm = customtkinter.CTkLabel(master=self.frame, text="RVC:Voice Model", font=("Impact", 18), text_color="deep sky blue")
        self.label_applio_vm.grid(row=9, column=3, pady=3, padx=6)
        self.entry_applio_vm = customtkinter.CTkEntry(master=self.frame, placeholder_text=r"logs\model\model.pth")
        self.entry_applio_vm.grid(row=10, column=3, pady=3, padx=6)
        #self.entry_applio_vm.insert(0, "") #Set your default voice model here
        self.applio_voice_model = ""
        #RVC INDEX FILE
        self.label_applio_if = customtkinter.CTkLabel(master=self.frame, text="RVC:Index File", font=("Impact", 18), text_color="deep sky blue")
        self.label_applio_if.grid(row=11, column=3, pady=3, padx=6)
        self.entry_applio_if = customtkinter.CTkEntry(master=self.frame, placeholder_text=r"logs\model\model.index")
        self.entry_applio_if.grid(row=12, column=3, pady=3, padx=6)
        #self.entry_applio_if.insert(0, "") #Set your default index model here
        self.applio_index_file = ""
        #RVC PITCH
        self.label_applio_pitch = customtkinter.CTkLabel(master=self.frame, text="RVC Pitch: 0", font=("Impact", 18), text_color="deep sky blue")
        self.label_applio_pitch.grid(row=13, column=3, pady=3, padx=6)
        self.applio_pitch_slider = customtkinter.CTkSlider(self.frame, width=150, from_=-24, to=24, number_of_steps=48, command=self.applio_pitch_slider_update)
        self.applio_pitch_slider.grid(row=14, column=3, padx=3, pady=6)
        self.applio_pitch_slider.set(0.0)
        self.pitch = 0.0
        #APPLIO FILE SAVE TIMEOUT SLIDER
        self.appliotimeout_slider = customtkinter.CTkSlider(self.frame, width=150, from_=10, to=360, number_of_steps=70, command=self.applio_timeout_slider_update)
        self.appliotimeout_slider.grid(row=15, column=3, padx=3, pady=6)
        self.appliotimeout_slider.set(0.15)
        self.applio_timeout = 15

        #4th column Whisper settings--------------------------------------------------------
        #VOSK RADIO BUTTON
        self.vosk_radio_button = customtkinter.CTkRadioButton(master=self.frame, variable=self.stt_radio_var, value=0, text="Vosk STT", font=("Impact", 14), text_color="blueviolet")
        self.vosk_radio_button.grid(row=1, column=4, pady=3, padx=6)
        #WHISPER RADIO BUTTON
        self.whisper_radio_button = customtkinter.CTkRadioButton(master=self.frame, variable=self.stt_radio_var, value=1, text="Whisper STT", font=("Impact", 14), text_color="blueviolet")
        self.whisper_radio_button.grid(row=2, column=4, pady=3, padx=6)
        #VOSK MODEL CFG
        self.labelvosk = customtkinter.CTkLabel(master=self.frame, text="Vosk Model", font=("Impact", 18), text_color="deep sky blue")
        self.labelvosk.grid(row=3, column=4, pady=3, padx=6)
        self.entryvosk = customtkinter.CTkEntry(master=self.frame, placeholder_text="vosk-model-small-en-us-0.15")
        self.entryvosk.insert(0, "vosk-model-small-en-us-0.15")
        self.entryvosk.grid(row=4, column=4, pady=3, padx=6)
        #WHISPER MODEL CFG
        self.label_whisper_model = customtkinter.CTkLabel(master=self.frame, text="Whisper Model", font=("Impact", 18), text_color="deep sky blue")
        self.label_whisper_model.grid(row=5, column=4, pady=3, padx=6)
        self.entry_whisper_model= customtkinter.CTkEntry(master=self.frame, placeholder_text="tiny, small, base, medium, large")
        self.entry_whisper_model.insert(0, "tiny")
        self.entry_whisper_model.grid(row=6, column=4, pady=3, padx=6)
        self.whispermodelname = "tiny" #Default
        #WHISPER GPU OR CPU CFG
        self.label_cuda_cpu = customtkinter.CTkLabel(master=self.frame, text="CUDA-CPU", font=("Impact", 18), text_color="green")
        self.label_cuda_cpu.grid(row=7, column=4, pady=3, padx=6)
        self.cuda_cpu_switch = customtkinter.CTkSwitch(master=self.frame, text="cuda", font=("Impact", 18), text_color="green", onvalue="cpu", offvalue="cuda", command=self.cuda_cpu_switch_update)
        self.cuda_cpu_switch.grid(row=8, column=4, padx=3, pady=6)
        self.whispercudacpu = "cuda"
        #WHISPER SILENCE THRESHOLD
        self.label_st = customtkinter.CTkLabel(master=self.frame, text="Silence Threshold", font=("Impact", 18), text_color="deep sky blue")
        self.label_st.grid(row=9, column=4, pady=3, padx=6)
        self.label_chunk_vol = customtkinter.CTkLabel(master=self.frame, text="0.0", font=("Impact", 18), text_color="dodgerblue")
        self.label_chunk_vol.grid(row=10, column=4, pady=3, padx=6)
        self.entry_st= customtkinter.CTkEntry(master=self.frame, placeholder_text="0.0005 default")
        self.entry_st.insert(0, 0.0005)
        self.entry_st.grid(row=11, column=4, pady=3, padx=6)
        self.silence_threshold = 0.0005
        #WHISPER AMOUNT OF SILENT CHUNKS BEFORE TRANSCRIBE
        self.label_silent_chunks = customtkinter.CTkLabel(master=self.frame, text="Min Silent Chunks", font=("Impact", 18), text_color="deep sky blue")
        self.label_silent_chunks.grid(row=12, column=4, pady=3, padx=6)
        self.entry_silent_chunks= customtkinter.CTkEntry(master=self.frame, placeholder_text="5 default")
        self.entry_silent_chunks.insert(0, 5)
        self.entry_silent_chunks.grid(row=13, column=4, pady=3, padx=6)
        self.minsilentchunks = 5
        #WHISPER DELAY FOR UNLOADING MODEL AND SHUTDOWN
        self.autotune_switch = customtkinter.CTkSwitch(master=self.frame, text="Autotune OFF", font=("Impact", 18), text_color="green", onvalue="True", offvalue="False", command=self.autotune_switch_update)
        self.autotune_switch.grid(row=14, column=4, padx=3, pady=6)
        self.autotune = "OFF"
        #LANGUAGE SETTINGS DROPDOWN BOX
        self.language_optionmenu = customtkinter.CTkOptionMenu(self.frame, values=["None", "en", "es", "fr", "de", "it", "ja", "ko", "zh-CN", "ar", "hi", "ru", "pt", "pl", "tr", "nl", "cs", "hu"], command=self.change_language_event)
        self.language_optionmenu.grid(row=15, column=4, padx=20, pady=(10, 20))
        self.language_optionmenu.set("None")
        self.whisperlang = None

        #BIG ENTRY FIELD FOR TEXT PLAYBACK
        self.entry_big= customtkinter.CTkEntry(master=self.frame, placeholder_text="Paste TEXT here for playback")
        self.entry_big.grid(row=17, rowspan=1, column=1, columnspan=4, pady=3, padx=6, sticky="nsew")
        self.playback_button = customtkinter.CTkButton(master=self.frame, hover=False, fg_color="green", text="Playback", font=("Impact", 18), text_color="white", command=self.playback_start)
        self.playback_button.grid(row=18, column=2, columnspan=2, pady=6, padx=6)

        #Validate function to check for numeric input
        def validate_numeric(P):
            if P == "" or (P.isdigit() or P.replace('.', '', 1).isdigit()):
                return True
            else:
                return False
        self.entrymt.configure(validate="key", validatecommand=(self.entrymt.register(validate_numeric), '%P'))
        self.entry_st.configure(validate="key", validatecommand=(self.entry_st.register(validate_numeric), '%P'))
        self.entry_silent_chunks.configure(validate="key", validatecommand=(self.entry_silent_chunks.register(validate_numeric), '%P'))

############___GUI_EVENTS___##################
#1)FIRST_COLUMN-----------------------------------------------------------------------------------------------
    #BTTS VOICE ID
    def btts_voiceid_slider_update(self, value):
        self.btts_voiceid_slider.configure(state="disabled")
        self.labelbttsvoiceid.configure(text=f"BTTS Voiceid: {int(float(value))}") # Update the label with the current slider value
        try:
            basictts.setProperty('voice',voices[int(value)].id)  # Set voice id
            basictts.say(("Voice " + str(int(float(value)))))
            basictts.runAndWait()
            self.btts_voiceid_slider.configure(state="normal")
        except:
            print("!!!!!!!!Voice setting out of range!!!!!!!!!!!")
            basictts.say(("Voice " + str(int(float(value))) + " out of range"))
            basictts.runAndWait()
            self.btts_voiceid_slider.configure(state="normal")
    #BTTS RATE
    def btts_rate_slider_update(self, value):
        self.labelbttsrate.configure(text=f"BTTS Rate: {int(float(value))}") # Update the label with the current slider value
        basictts.setProperty('rate', int(value))
    #BTTS VOLUME
    def btts_volume_slider_update(self, value):
        self.labelbttsvol.configure(text=f"BTTS Vol: {value:.1f}") # Update the label with the current slider value
        basictts.setProperty('volume', value)
    #CHANGE GUI SIZE
    def change_scaling_event(self, new_scaling: str):
        new_scaling_float = int(new_scaling.replace("%", "")) / 100
        customtkinter.set_widget_scaling(new_scaling_float)
    #Confirm ON or OFF
    def confirm_on_off_switch_update(self):
        self.confirm_on_off = self.confirm_on_off_switch.get()
        print(self.confirm_on_off)
        self.confirm_on_off_switch.configure(text=self.confirm_on_off)
#2)SECOND_COLUMN----------------------------------------------------------------------------------------------
    #LANGUAGE MODEL TEMPERATURE
    def lm_tempurature_slider_update(self, value):
        self.labeltemp.configure(text=f"Temp: {value:.1f}") # Update the label with the current slider value
        self.lm_temp = value
    #LANGUAGE MODEL FREQUENCY PENALTY
    def lm_frequency_penalty_slider_update(self, value):
        self.labelfp.configure(text=f"FreqPenalty: {value:.1f}") # Update the label with the current slider value
        self.lm_freq_p_val = value
    #LANGUAGE MODEL PRESENCE PENALTY
    def lm_presence_penalty_slider_update(self, value):
        self.labelpp.configure(text=f"PresPenalty: {value:.1f}") # Update the label with the current slider value
        self.lm_presence_p_val = value
#3)THIRD_COLUMN-----------------------------------------------------------------------------------------------
    #APPLIO RATE
    def applio_rate_slider_update(self, value):
        self.label_applio_rate.configure(text=f"TTS Rate: {value:.1f}") # Update the label with the current slider value
        self.applio_rate = value
    #APPLIO PITCH
    def applio_pitch_slider_update(self, value):
        self.label_applio_pitch.configure(text=f"RVC Pitch: {value:.1f}") # Update the label with the current slider value
        self.pitch = value
    #APPLIO TIMEOUT SLIDER
    def applio_timeout_slider_update(self, value):
        self.labelappliotimeout.configure(text=f"Applio Save Timeout: {value:.1f}") # Update the label with the current slider value
        self.applio_timeout = value
#4)FORTH_COLUMN-----------------------------------------------------------------------------------------------
    #CUDA OR CPU SWITCH
    def cuda_cpu_switch_update(self):
        self.whispercudacpu = self.cuda_cpu_switch.get()
        print(self.whispercudacpu)
        self.cuda_cpu_switch.configure(text=self.whispercudacpu)
    #AUTOTUNE SWITCH
    def autotune_switch_update(self):
        self.autotune = self.autotune_switch.get()
        if self.autotune == "True":
            print(self.autotune)
            self.autotune_switch.configure(text="Autotune ON")
        if self.autotune == "False":
            print(self.autotune)
            self.autotune_switch.configure(text="Autotune OFF")
    #CHANGE LANGUAGE EVENT
    def change_language_event(self, value):
        self.whisperlang = value
        if self.whisperlang == "None":
            self.whisperlang = None
        print(self.whisperlang)

#########################################################################################___BUTTON_EVENTS___################################################################################################
    def button_start(self):
        self.start_button.configure(fg_color="red", text="Stop", command=self.button_stop)
        print("Start Button Activated")
        #Disable GUI-------------FIRST COLUMN------------------------------------------
        self.btts_radio_button.configure(state="disabled")
        self.chat_history_button.configure(state="disabled")
        self.btts_voiceid_slider.configure(state="disabled")
        self.btts_rate_slider.configure(state="disabled")
        self.bttsvol_slider.configure(state="disabled")
        self.stream_onoff_button.configure(state="disabled")
        self.entrynote.configure(state="disabled")
        self.scaling_optionemenu.configure(state="disabled")
        #Disable GUI-------------SECOND COLUMN-----------------------------------------
        self.entrykw.configure(state="disabled")
        self.entryun.configure(state="disabled")
        self.entrymt.configure(state="disabled")
        self.tempurature_slider.configure(state="disabled")
        self.freqpen_slider.configure(state="disabled")
        self.prespen_slider.configure(state="disabled")
        #Disable GUI-------------THIRD COLUMN------------------------------------------
        self.applio_radio_button.configure(state="disabled")
        self.ttsrvc_button.configure(state="disabled")
        self.entry_applio_folder.configure(state="disabled")
        self.entry_applio_ttsv.configure(state="disabled")
        self.applio_rate_slider.configure(state="disabled")
        self.entry_applio_vm.configure(state="disabled")
        self.entry_applio_if.configure(state="disabled")
        self.applio_pitch_slider.configure(state="disabled")
        self.label_applio_pitch.configure(state="disabled")
        #Disable GUI-------------FORTH COLUMN------------------------------------------
        self.vosk_radio_button.configure(state="disabled")
        self.whisper_radio_button.configure(state="disabled")
        self.entryvosk.configure(state="disabled")
        self.entry_whisper_model.configure(state="disabled")
        self.cuda_cpu_switch.configure(state="disabled")
        self.entry_st.configure(state="disabled")
        self.entry_silent_chunks.configure(state="disabled")
        self.autotune_switch.configure(state="disabled")
        self.language_optionmenu.configure(state="disabled")
        #------------------------------------------------------------------------------
        self.playback_button.configure(state="disabled")
        self.startstop = "ON"
        self.main_loop_thread = threading.Thread(target=main_loop)
        self.main_loop_thread.start()
        return 

    def button_stop(self):
        print("Stop Button Activated")
        self.startstop = "OFF"
        time.sleep(systemdelay)
        #Enable GUI-------------FIRST COLUMN------------------------------------------
        self.btts_radio_button.configure(state="normal")
        self.chat_history_button.configure(state="normal")
        self.btts_voiceid_slider.configure(state="normal")
        self.btts_rate_slider.configure(state="normal")
        self.bttsvol_slider.configure(state="normal")
        self.stream_onoff_button.configure(state="normal")
        self.entrynote.configure(state="normal")
        self.scaling_optionemenu.configure(state="normal")
        #Enable GUI-------------SECOND COLUMN-----------------------------------------
        self.entrykw.configure(state="normal")
        self.entryun.configure(state="normal")
        self.entrymt.configure(state="normal")
        self.tempurature_slider.configure(state="normal")
        self.freqpen_slider.configure(state="normal")
        self.prespen_slider.configure(state="normal")
        #Enable GUI-------------THIRD COLUMN------------------------------------------
        self.applio_radio_button.configure(state="normal")
        self.ttsrvc_button.configure(state="normal")
        self.entry_applio_folder.configure(state="normal")
        self.entry_applio_ttsv.configure(state="normal")
        self.applio_rate_slider.configure(state="normal")
        self.entry_applio_vm.configure(state="normal")
        self.entry_applio_if.configure(state="normal")
        self.applio_pitch_slider.configure(state="normal")
        self.label_applio_pitch.configure(state="normal")
        #Enable GUI-------------FORTH COLUMN------------------------------------------
        self.vosk_radio_button.configure(state="normal")
        self.whisper_radio_button.configure(state="normal")
        self.entryvosk.configure(state="normal")
        self.entry_whisper_model.configure(state="normal")
        self.cuda_cpu_switch.configure(state="normal")
        self.entry_st.configure(state="normal")
        self.entry_silent_chunks.configure(state="normal")
        self.autotune_switch.configure(state="normal")
        self.language_optionmenu.configure(state="normal")
        #-----------------------------------------------------------------------------
        self.playback_button.configure(state="normal")
        self.start_button.configure(fg_color="green",text="Start", command=self.button_start)     
        return
    
    #PLAYBACK APPLIO BASIC TTS
    def play_tts(self):
        self.ttsrvc_button.configure(fg_color="deep sky blue",text="Applio TTS", command=self.play_rvc)
        self.ttsrvc = "TTS"
        print("TTS button selected")
        return
    #PLAYBACK APPLIO RVC
    def play_rvc(self):
        self.ttsrvc_button.configure(fg_color="blue", text="Applio RVC", command=self.play_tts)
        self.ttsrvc = "RVC"
        print("RVC button selected")
        return
    #LMSTUDIO CHAT HISTORY ON
    def button_chat_history_on(self):
        self.chat_history_button.configure(fg_color="deep sky blue", text="Chat History ON", command=self.button_chat_history_off)
        self.chat_history = "ON"
        print("Chat history ON")
        return
    #LMSTUDIO CHAT HISTORY OFF
    def button_chat_history_off(self):
        global history
        self.chat_history_button.configure(fg_color="blue",text="Chat History OFF", command=self.button_chat_history_on)
        self.chat_history = "OFF"
        self.history = [{"role": "system", "content": system_message}]
        print("Chat History OFF. History Erased")
        return
    #FASTER BTTS ON
    def button_stream_on(self):
        self.stream_onoff_button.configure(fg_color="deep sky blue", text="Stream ON", command=self.button_stream_off)
        self.btts_stream_on = True
        self.tts_radio_var.set(0)
        self.tts_radiovar = self.tts_radio_var.get()
        print("Stream ON: faster response while using Basic TTS. Stream mode deactivated")
        return
    #FASTER BTTS OFF
    def button_stream_off(self):
        self.stream_onoff_button.configure(fg_color="blue",text="Stream OFF", command=self.button_stream_on)
        self.btts_stream_on = False
        print("Stream OFF")
        return
    #APPLIO NOT FASTER BTTS
    def applio_not_stream(self):
        self.tts_radiovar = self.tts_radio_var.get()
        if self.tts_radiovar == 1:
            self.button_stream_off()
            print("Applio activated. Stream mode will be deactivated")

    def playback(self):
        if interruptbigtext == False:
            time.sleep(0.2)
            for x in self.big_entry_text_split:
                if interruptbigtext == False:
                    basictts.say(x)
                    print(x)
                    basictts.runAndWait()
            else:
                self.start_button.configure(state="normal")
                self.playback_button.configure(fg_color="green", text="Playback", command=self.playback_start)
                print("Ending Playback")

    def playback_start(self):
        global interruptbigtext, playback_thread
        self.start_button.configure(state="disabled")
        self.playback_button.configure(fg_color="red",text="Stop", command=self.playback_stop)
        interruptbigtext = False
        self.big_entry_text = self.entry_big.get()
        self.big_entry_text = self.big_entry_text.replace('Mr.', 'Mr').replace('Mrs.', 'Mrs').replace('<|eot_id|>', '').replace('*', ' ').replace('\n', '.').replace('\t', ' ').replace('<|start_header_id|>', '').replace('<|end_header_id|>', '').replace('user', '') #.replace(',', '.')
        self.big_entry_text_split = self.big_entry_text.split('.')
        print("Starting Playback")
        playback_thread = threading.Thread(target=self.playback)
        playback_thread.start()
        time.sleep(0.2)

    def playback_stop(self):
        global interruptbigtext
        self.playback_button.configure(fg_color="green", text="Playback", command=self.playback_start)
        interruptbigtext = True
        time.sleep(1)
        self.start_button.configure(state="normal")

if __name__ == "__main__":
    app = App()
    app.mainloop()