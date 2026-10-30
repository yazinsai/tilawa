"""Independent literal-ASR calibration. No Quran reference or benchmark labels sent to models."""
import hashlib,json
from pathlib import Path
import modal
ROOT=Path(__file__).resolve().parent.parent
cache=modal.Volume.from_name('tilawa-hf-cache')
app=modal.App('tilawa-review-asr')
base=modal.Image.debian_slim(python_version='3.12').apt_install('ffmpeg').env({'HF_HOME':'/cache/hf'})
qimage=base.pip_install('qwen-asr','soundfile')
wimage=base.pip_install('faster-whisper','soundfile')
@app.function(image=qimage,gpu='L4',volumes={'/cache':cache},timeout=1200,max_containers=1)
def qwen(clips):
 import torch,tempfile,importlib.metadata
 from qwen_asr import Qwen3ASRModel
 model=Qwen3ASRModel.from_pretrained('Qwen/Qwen3-ASR-1.7B',dtype=torch.bfloat16,device_map='cuda:0',max_inference_batch_size=1,max_new_tokens=512)
 out=[]
 for cid,data in clips:
  with tempfile.NamedTemporaryFile(suffix='.wav') as f:
   f.write(data);f.flush();r=model.transcribe(audio=f.name,language='Arabic')[0]
   out.append({'clip':cid,'audioSha256':hashlib.sha256(data).hexdigest(),'text':r.text,'language':r.language})
 return {'model':'Qwen/Qwen3-ASR-1.7B','packageVersion':importlib.metadata.version('qwen-asr'),'referencePrompt':None,'results':out}
@app.function(image=wimage,cpu=8,volumes={'/cache':cache},timeout=1200,max_containers=1)
def whisper(clips):
 import tempfile,importlib.metadata
 from faster_whisper import WhisperModel
 model=WhisperModel('large-v3',device='cpu',compute_type='int8',cpu_threads=8,download_root='/cache/whisper')
 out=[]
 for cid,data in clips:
  with tempfile.NamedTemporaryFile(suffix='.wav') as f:
   f.write(data);f.flush();import soundfile as sf
   import numpy as np
   audio,sr=sf.read(f.name,dtype='float32');assert sr==16000
   if audio.ndim>1:audio=audio.mean(axis=1)
   segments,info=model.transcribe(np.asarray(audio),language='ar',beam_size=5,condition_on_previous_text=False,vad_filter=False)
   out.append({'clip':cid,'audioSha256':hashlib.sha256(data).hexdigest(),'segments':[{'start':s.start,'end':s.end,'text':s.text,'avgLogprob':s.avg_logprob} for s in segments]})
 return {'model':'large-v3','implementation':'faster-whisper','packageVersion':importlib.metadata.version('faster-whisper'),'referencePrompt':None,'results':out}
@app.local_entrypoint()
def main():
 dest=ROOT/'.scratch/help-review/asr';dest.mkdir(parents=True,exist_ok=True)
 ids=list(dict.fromkeys(f['clip'] for f in json.loads((ROOT/'.scratch/help-review/human-findings.json').read_text())['findings']))
 clips=[(cid,(ROOT/'.scratch/help/wav'/f'{cid}.wav').read_bytes()) for cid in ids]
 for name,fn in [('qwen3',qwen),('whisper-large-v3',whisper)]:
  p=dest/(name+'.json')
  if p.exists():continue
  print('Calibrating',name,'on',len(clips),'reviewed clips',flush=True)
  r=fn.remote(clips);p.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n');print('Saved',p.name,flush=True)
