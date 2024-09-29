# instantiate the pipeline
from pyannote.audio import Pipeline
pipeline = Pipeline.from_pretrained(
  "pyannote/speaker-diarization-3.1",
  use_auth_token="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")

# run the pipeline on an audio file
diarization = pipeline("data/asr_speaker_demo.wav")

print(diarization)
# # dump the diarization output to disk using RTTM format
# with open("result/agora.rttm", "w") as rttm:
#     diarization.write_rttm(rttm)
