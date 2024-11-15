import whispercpp as w
import time

# Start timer
start_time = time.time()

transcriber = w.Whisper.from_pretrained("small")
res = transcriber.transcribe_from_file("audio.wav")
print(res)

# End timer
end_time = time.time()

# Calculate elapsed time
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.2f} seconds")

