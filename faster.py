from faster_whisper import WhisperModel

# Set the model size to "small" for lighter resource usage
model_size = "small"

# Initialize the model
# This example uses CPU with INT8 to optimize speed; you can change to "cuda" and "float16" if you have a compatible GPU
model = WhisperModel(model_size, device="cpu", compute_type="int8")

# Transcribe the audio file with a smaller beam size to optimize speed
segments, info = model.transcribe("audio.wav", beam_size=3)

# Output the detected language and probability
print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

# Print each transcription segment
for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))

