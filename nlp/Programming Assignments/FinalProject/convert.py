from pydub import AudioSegment
import pandas as pd
import os

# Convert mp3 to wav for pytorch ingestion
for root, dirs, files in os.walk("C:\\Users\\EricL\\School\\nlp\\Programming Assignments\\FinalProject\\data\\archive\\"):
    for name in files:
        no_ext_name, ext = name.split(sep=".")
        if ext == "csv" and (no_ext_name == "cv-valid-test" or no_ext_name == "cv-valid-train"):
            df = pd.read_csv(os.path.join(root, name))
            for idx, row in df.iterrows():
                # convert mp3 to wav
                src = os.path.join(root, no_ext_name, row["filename"])
                dst = os.path.join(root, no_ext_name, row["filename"].split(sep=".")[0]) + ".wav"
                sound = AudioSegment.from_mp3(src)
                sound.export(dst, format="wav")
