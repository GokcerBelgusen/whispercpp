import whispercpp as w

transcriber = w.Whisper.from_pretrained("small")
res = transcriber.transcribe_from_file("audio.wav")
print(res)

