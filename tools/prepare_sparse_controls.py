"""Extract untouched contiguous contexts around independently checked self-corrections."""
import hashlib,json,wave
from pathlib import Path
root=Path('/lab/data/sparse-real-controls');root.mkdir(exist_ok=True)
rows=[]
for sid,start,end,surah,first,last,at,attempt,correction in [
 ('rec-yazin-rec-v1-01-t1',125.8,140.,96,1,3,4,(132.61,132.79),(134.53,134.77)),
 ('rec-yazin-rec-v1-02-t1',52.,81.,71,6,8,6,(64.29,64.77),(66.03,66.51))]:
 src=Path('/lab/data/recordings/sessions')/(sid+'.phone.wav')
 with wave.open(str(src)) as w:
  params=w.getparams();w.setpos(round(start*w.getframerate()));data=w.readframes(round((end-start)*w.getframerate()))
 dest=root/(sid+'.wav')
 with wave.open(str(dest),'wb') as w:w.setparams(params);w.writeframes(data)
 rows.append({'id':sid,'path':str(dest),'surah':surah,'fromAyah':first,'toAyah':last,'location':{'surah':surah,'ayah':2 if surah==96 else 7,'word':at},'sourceStartS':start,'attempt':[a-start for a in attempt],'correction':[a-start for a in correction],'sha256':hashlib.sha256(dest.read_bytes()).hexdigest()})
(root/'controls.json').write_text(json.dumps(rows,indent=1));print(json.dumps(rows))
