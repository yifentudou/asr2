# instantiate the pipeline
from pyannote.audio import Pipeline
import pickle
pipeline = Pipeline.from_pretrained(
  "pyannote/speaker-diarization-3.1",
  use_auth_token="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")

# run the pipeline on an audio file
diarization = pipeline("data/asr_speaker_demo.wav")

# print(diarization)
# dump the diarization output to disk using RTTM format
# with open("result/agora.rttm", "w") as rttm:
#     diarization.write_rttm(rttm)
dialogue = []
with open("result/agora.rttm", 'wb') as f:
    pickle.dump(dialogue, f)
end_time = time.time()
print(file + " spend time:" + str(end_time - start_time))      
