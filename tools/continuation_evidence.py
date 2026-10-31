import json,os,runpy
from pathlib import Path
rows=json.loads(Path('/app/tools/continuation-sites.json').read_text())
row=rows[int(os.environ['LAB_SHARD'])]
models=dict(main='spliced-clean-int8',uniform='phone-channel-int8',targeted='targeted-consonants-int8',selected='sparse-consonants-v2-int8',random='sparse-random-v2-int8')
(Path(os.environ['LAB_OUT'])/'engine.json').write_text(json.dumps(dict(row=row,results={k:dict(model=v) for k,v in models.items()})))
runpy.run_path('/app/tools/continuation_trace.py')
