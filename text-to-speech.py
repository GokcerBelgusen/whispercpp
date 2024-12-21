from transformers import VitsModel, AutoTokenizer
import torch
import numpy as np
import pyaudio
 

# Load the model and tokenizer
model = VitsModel.from_pretrained("facebook/mms-tts-tur")
tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-tur")

# Text to be synthesized
text = "Kara delik, çok güçlü bir yerçekimine sahip olan, ışığın bile kaçamadığı kozmik cisimlerdir. Büyük bir kütlenin çok küçük bir alana sıkışmasıyla oluşurlar. Genellikle devasa yıldızların ömrünün sonuna geldiğinde çökmesiyle meydana gelirler. Kara deliklerin olay ufku adı verilen bir sınırları vardır; bu sınırdan içeri giren hiçbir şey, ışık dahi, dışarı çıkamaz. Kara delikler, evrenin en gizemli ve en yoğun nesnelerinden biridir."

# Tokenize the text input
inputs = tokenizer(text, return_tensors="pt")

torch.set_num_threads(4)

# Generate the waveform using the model
with torch.no_grad():
    output = model(**inputs).waveform

# Convert to NumPy array and ensure it's in the correct range
output_np = output.cpu().numpy()

# Ensure the data is in the range [-1.0, 1.0]
output_np = np.clip(output_np, -1.0, 1.0)

# Convert to int16 for real-time audio playback
output_int16 = (output_np * 32767).astype(np.int16)

# Initialize PyAudio
p = pyaudio.PyAudio()

# Open a PyAudio stream
stream = p.open(
    format=pyaudio.paInt16,
    channels=1,  # Mono audio
    rate=model.config.sampling_rate,
    output=True
)

# Play the audio data in chunks for real-time playback
chunk_size = 1024
for i in range(0, len(output_int16), chunk_size):
    chunk = output_int16[i:i + chunk_size]
    stream.write(chunk.tobytes())

# Stop and close the stream
stream.stop_stream()
stream.close()

# Terminate PyAudio
p.terminate()

