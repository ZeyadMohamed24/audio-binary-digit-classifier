import pyaudio
import wave
import os

def record_voice(
    output_filename, record_seconds=2, sample_rate=44100, channels=1, chunk_size=1024
):
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    audio = pyaudio.PyAudio()
    stream = audio.open(
        format=pyaudio.paInt16,
        channels=channels,
        rate=sample_rate,
        input=True,
        frames_per_buffer=chunk_size,
    )
    print("Recording...")
    frames = []
    for i in range(0, int(sample_rate / chunk_size * record_seconds)):
        data = stream.read(chunk_size)
        frames.append(data)
    print("Recording finished.")
    stream.stop_stream()
    stream.close()
    audio.terminate()
    with wave.open(output_filename, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
        wf.setframerate(sample_rate)
        wf.writeframes(b"".join(frames))
    print(f"File saved as {output_filename}")
