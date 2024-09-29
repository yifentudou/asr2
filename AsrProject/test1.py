from pyannote.audio import Pipeline
import utils

# 创建对应pipeline管道模型，调用预训练模型，这里是指定了调用模型的相关路径。
pipeline = Pipeline.from_pretrained(r"speaker-diarization-3.1/config.yaml")
# run the pipeline on an audio file
diarization = pipeline("data/asr_speaker_demo.wav")

# dump the diarization output to disk using RTTM format
with open("audio1.rttm", "wb") as rttm:
    diarization.write_rttm(rttm)
    
