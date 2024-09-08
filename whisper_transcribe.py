import pyaudio  
import numpy as np  
import whisper  
import time
import torch  # Import torch for GPU support
import sys
#This is my attempt at running the whisper transcribe process as a subprocess to get rid of the zombie thread that will not shut down when run.
#This worked in VS code but not in the venv so I ditched it when I realized whisper was only creating one extra thread, not multiples which doesn't seem to effect the overall function.
print("------------------------(whisper_transcrip.py)Started--------------------------")   
try:
    # Get command-line arguments
    device = sys.argv[1] if len(sys.argv) > 1 else "cpu"
    silence_threshold = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0003
    whispermodelname = sys.argv[3] if len(sys.argv) > 3 else "tiny"
    min_silent_chunks = sys.argv[4] if len(sys.argv) > 4 else 5
    min_silent_chunks = int(float(min_silent_chunks))
    # Load model on appropriate device  
    if torch.cuda.is_available():
        device = device
    else:
        print("Cuda unavailable. Using cpu.")
        device = "cpu"
    print("Device: ", device)
    whispermodel = whisper.load_model(whispermodelname, in_memory=True).to(device)  # Move the model to GPU
    whispermodel.eval() # Improve performance
    print("Model is on:", next(whispermodel.parameters()).device)  # Should print 'cuda:0' if on GPU
except ValueError as ve:
    print(ve)
def transcribe():
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
                        frames_per_buffer=4096) #8192 4096 2048 keep these numbers equal or you might get buffer overflow
        frames = []
    except ValueError as ve:
        print("Error:", ve, "ini np, pyaudio, torch")
    try:
        while True:
            try:
                # Read audio data from the stream  
                data = stream.read(4096) #8192 4096 2048 keep these numbers equal or you might get buffer overflow
                audio_chunk = np.frombuffer(data, dtype=np.float32)
                audio_mean = np.abs(audio_chunk).mean()

                # Output audio mean
                print("audio_mean:", audio_mean) #------------------OUTPUT---------------
                # Check if the audio level exceeds the silence threshold  
            except ValueError as ve:
                print(ve, "readstream and get mean")
            if audio_mean > silence_threshold:
                frames.append(data)
                silent_chunks = 0  # Reset silent chunks if sound is detected  
            else:
                silent_chunks += 1  # Increment silent chunk count

            # If silence is detected for 'min_silent_chunks' chunks, process the recorded audio  
            if silent_chunks >= min_silent_chunks:
                if frames:  # Only process if there are frames recorded  
                    audio_data = b''.join(frames)  # Concatenate collected frames  
                    frames = []  # Clear the list for the next batch

                    # Convert byte data to NumPy array  
                    audio_tensor = np.frombuffer(audio_data, dtype=np.float32).copy()

                    # Perform speech recognition directly on the audio tensor  
                    result = whispermodel.transcribe(audio_tensor)

                    # Print the recognized text if it's not empty  
                    if result['text'].strip():
                        try:
                            print("Transcription:", result['text'].strip()) #---------------------OUTPUT-----------------------
                        except ValueError as ve:
                            print("Error:", ve)
                    else:
                        print("")

                silent_chunks = 0  # Reset silence counter to start a new listening session

            time.sleep(0.1)  # Adjust this delay as needed
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

if __name__ == "__main__":
    transcribe()