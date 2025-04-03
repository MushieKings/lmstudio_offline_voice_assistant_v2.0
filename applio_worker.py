import sys
import asyncio
from gradio_client import Client, handle_file

async def main():
    # Read command-line arguments
    model_response = sys.argv[1]
    applio_tts_voice = sys.argv[2]
    applio_rate = int(float(sys.argv[3]))
    pitchvar = int(float(sys.argv[4]))
    applio_folder = sys.argv[5]
    applio_voice_model = sys.argv[6]
    applio_index_file = sys.argv[7]
    ttsrvc = sys.argv[8]
    autotune = sys.argv[9]
    tts_output_folder = (applio_folder + "/assets/audios/tts_output.wav").replace('/', '\\')
    rvc_output_folder = (applio_folder + "/assets/audios/tts_rvc_output.wav").replace('/', '\\')

    if autotune == "True":
        autotune = True
    if autotune == "False":
        autotune = False

    try:
        client = Client("http://127.0.0.1:6969/") #Change port if needed
    except Exception as e:
        print("Unable to connect to applio:", e)
        sys.exit(1)

    try:
        if ttsrvc == "TTS":
            result = client.predict(
                tts_text=model_response,
                tts_voice=applio_tts_voice,
                tts_rate=applio_rate,
                pitch=pitchvar,
#                filter_radius=3, # variable not found
                index_rate=0.75,
                volume_envelope=1,
                protect=0.5,
                hop_length=128,
                f0_method="rmvpe",
                output_tts_path=tts_output_folder,
                output_rvc_path=rvc_output_folder,
                pth_path=None,
                index_path=None,
                split_audio=False,
                f0_autotune=autotune,
                f0_autotune_strength=1,
                clean_audio=True,
                clean_strength=0.5,
                export_format="WAV",
                f0_file=None,
                embedder_model="contentvec",
                embedder_model_custom=None,
#                upscale_audio=False, # variable not found
#                api_name="/run_tts_script" # variable not found
            )

        elif ttsrvc == "RVC":
            result = client.predict(
                tts_text=model_response,
                tts_voice=applio_tts_voice,
                tts_rate=applio_rate,
                pitch=pitchvar,
#                filter_radius=3,
                index_rate=0.75,
                volume_envelope=1,
                protect=0.5,
                hop_length=128,
                f0_method="rmvpe",
                output_tts_path=tts_output_folder,
                output_rvc_path=rvc_output_folder,
                pth_path=applio_voice_model,
                index_path=applio_index_file,
                split_audio=False,
                f0_autotune=autotune,
                f0_autotune_strength=1,
                clean_audio=True,
                clean_strength=0.5,
                export_format="WAV",
                f0_file=None,
                embedder_model="contentvec",
                embedder_model_custom=None,
#                upscale_audio=False,
#                api_name="/run_tts_script"
            )

    except ValueError as ve:
        print("APPLIO ERROR:", ve)
    except Exception as e:
        print(f"APPLIO ERROR: {e}")
    finally:
        client.close()
        print("result: ", result)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except RuntimeError as e:
        print(f"Runtime error: {e}")

#Args:
#    audio_input_path (str): Path to the input audio file.
#    audio_output_path (str): Path to the output audio file.
#    model_path (str): Path to the voice conversion model.
#    index_path (str): Path to the index file.
#    sid (int, optional): Speaker ID. Default is 0.
#    pitch (str, optional): Key for F0 up-sampling. Default is None.
#    f0_file (str, optional): Path to the F0 file. Default is None.
#    f0_method (str, optional): Method for F0 extraction. Default is None.
#    index_rate (float, optional): Rate for index matching. Default is None.
#    resample_sr (int, optional): Resample sampling rate. Default is 0.
#    volume_envelope (float, optional): RMS mix rate. Default is None.
#    protect (float, optional): Protection rate for certain audio segments. Default is None.
#    hop_length (int, optional): Hop length for audio processing. Default is None.
#    split_audio (bool, optional): Whether to split the audio for processing. Default is False.
#    f0_autotune (bool, optional): Whether to use F0 autotune. Default is False.
#    filter_radius (int, optional): Radius for filtering. Default is None.
#    embedder_model (str, optional): Path to the embedder model. Default is None.
#    embedder_model_custom (str, optional): Path to the custom embedder model. Default is None.
#    clean_audio (bool, optional): Whether to clean the audio. Default is False.
#    clean_strength (float, optional): Strength of the audio cleaning. Default is 0.7.
#    export_format (str, optional): Format for exporting the audio. Default is "WAV".
#    upscale_audio (bool, optional): Whether to upscale the audio. Default is False.
#    formant_shift (bool, optional): Whether to shift the formants. Default is False.
#    formant_qfrency (float, optional): Formant frequency. Default is 1.0.
#    formant_timbre (float, optional): Formant timbre. Default is 1.0.
#    reverb (bool, optional): Whether to apply reverb. Default is False.
#    pitch_shift (bool, optional): Whether to apply pitch shift. Default is False.
#    limiter (bool, optional): Whether to apply a limiter. Default is False.
#    gain (bool, optional): Whether to apply gain. Default is False.
#    distortion (bool, optional): Whether to apply distortion. Default is False.
#    chorus (bool, optional): Whether to apply chorus. Default is False.
#    bitcrush (bool, optional): Whether to apply bitcrush. Default is False.
#    clipping (bool, optional): Whether to apply clipping. Default is False.
#    compressor (bool, optional): Whether to apply a compressor. Default is False.
#    delay (bool, optional): Whether to apply delay. Default is False.
#    sliders (dict, optional): Dictionary of effect parameters. Default is None.
