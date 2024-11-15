from vosk import Model, KaldiRecognizer
import wave
import time

# Start timer
start_time = time.perf_counter()

wf = wave.open("audio-tr.wav", "rb")
model = Model("models/vosk-model-small-tr-0.3")  # Download a Vosk model and specify the path
rec = KaldiRecognizer(model, wf.getframerate())

while True:
    data = wf.readframes(4000)
    if len(data) == 0:
        break
    if rec.AcceptWaveform(data):
        print(rec.Result())

# End timer
end_time = time.perf_counter()

# Calculate elapsed time
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.6f} seconds")

