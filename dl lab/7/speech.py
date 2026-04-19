# Simple Speech to Text Note Taker

import speech_recognition as sr

r = sr.Recognizer()

with sr.Microphone() as source:
    print("Speak...")
    audio = r.listen(source)

text = r.recognize_google(audio)

file = open("notes.txt", "w")
file.write(text)
file.close()

print("Notes saved")