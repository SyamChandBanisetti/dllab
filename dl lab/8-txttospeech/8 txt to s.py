from gtts import gTTS

text = "Hello, this is a text to speech example."
tts = gTTS(text)

tts.save("sample_output1.mp3")
print("Audio saved as output.mp3")
