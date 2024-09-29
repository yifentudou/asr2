from pyannote.core import Segment
from pyannote.audio import Pipeline
import whisper
import pickle
import torch
import time
import os

 
from pyannote.audio import Pipeline
from pyannote.core import Annotation
file = "data/asr_speaker_demo.wav"
 
def get_text_with_timestamp(transcribe_res):
    print(transcribe_res)
    timestamp_texts = []
    for item in transcribe_res["segments"]:
        start = item["start"]
        end = item["end"]
        # text = convert(item["text"],'zh-cn').strip()
        text = item["text"]
        timestamp_texts.append((Segment(start, end), text))
    return timestamp_texts
 
 
def add_speaker_info_to_text(timestamp_texts, ann):
    spk_text = []
    for seg, text in timestamp_texts:
        print(ann.crop(seg))
        spk = ann.crop(seg).argmax()
        spk_text.append((seg, spk, text))
    # print("spk_text是：",spk_text)
    return spk_text
 
 
def merge_cache(text_cache):
    sentence = ''.join([item[-1] for item in text_cache])
    spk = text_cache[0][1]
    start = round(text_cache[0][0].start, 1)
    end = round(text_cache[-1][0].end, 1)
    return Segment(start, end), spk, sentence
 
 
PUNC_SENT_END = ['.', '?', '!', "。", "？", "！"]
 
 
def merge_sentence(spk_text):
    merged_spk_text = []
    pre_spk = None
    text_cache = []
    for seg, spk, text in spk_text:
        if spk != pre_spk and len(text_cache) > 0:
            merged_spk_text.append(merge_cache(text_cache))
            text_cache = [(seg, spk, text)]
            pre_spk = spk
        elif spk == pre_spk and text == text_cache[-1][2]:
            print(text_cache[-1][2])
            # print(text)
            continue
 
            # merged_spk_text.append(merge_cache(text_cache))
            # text_cache.append((seg, spk, text))
            # pre_spk = spk
        else:
            text_cache.append((seg, spk, text))
            pre_spk = spk
    if len(text_cache) > 0:
        merged_spk_text.append(merge_cache(text_cache))
    return merged_spk_text
 
 
def diarize_text(transcribe_res, diarization_result):
    timestamp_texts = get_text_with_timestamp(transcribe_res)
    spk_text = add_speaker_info_to_text(timestamp_texts, diarization_result)
    res_processed = merge_sentence(spk_text)
    # print("res_processeds是：",res_processed)
    # res_processed = spk_text
    return res_processed
 
 
 
# def write_to_txt(spk_sent, file):
#     with open(file, 'w') as fp:
#         for seg, spk, sentence in spk_sent:
#             line = f'{seg.start:.2f} {seg.end:.2f} {spk} {sentence}\n'
#             fp.write(line)
# def format_time(seconds):
#     # 计算小时、分钟和秒数
#     hours = seconds // 3600
#     minutes = (seconds % 3600) // 60
#     seconds = seconds % 60
 
#     # 格式化输出
#     return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
if __name__ == "__main__":
    speaker_diarization = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1",use_auth_token="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")
    speaker_diarization.to(torch.device("cpu"))
    asr_model=whisper.load_model("large")
    asr_model.to(torch.device("cpu"))

    start_time = time.time()
    print(file)
 
    dialogue_path = "audios_txt/" + file.split(".")[0] + ".pkl"
    audio = "audios_wav/" + file
    asr_result = asr_model.transcribe(audio,
                                        initial_prompt="随便")
    asr_time = time.time()
    print("ASR time:" + str(asr_time - start_time))
 
    diarization_result: Annotation = speaker_diarization(audio)
    final_result = diarize_text(asr_result, diarization_result)
 
 
    dialogue = []
    for segment, spk, sent in final_result:
        content = {'speaker': spk, 'start': segment.start, 'end': segment.end, 'text': sent}
        dialogue.append(content)
        # print("_______________________________")
        print("[%.2fs -> %.2fs] %s %s" % (segment.start, segment.end, spk,sent))
        end_time = time.time()
        print(file + " spend time:" + str(end_time - start_time))
 
    # with open(dialogue_path, 'wb') as f:
    #     pickle.dump(dialogue, f)
    # end_time = time.time()
    # print(file + " spend time:" + str(end_time - start_time))