"""Isolated reduced-size MPS backward comparison; never loads/saves checkpoints.

Run from the model repository with python3 tools/profile_backward.py.
The baseline is the committed pre-batched microscope; the full backward implementation and its diagnostic helpers are
substituted. Ten synthetic steps alternate reports; discard the first two.
This measures real forward/loss/backward/optimizer code at reduced dimensions,
not production-window throughput. No tutor, sensors, networking or bot startup.
"""
import sys,time,contextlib,io,cProfile,pstats,ast,subprocess,types,statistics
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch,config
torch.manual_seed(0)
config.embedDimension=32;config.numNeurons=64;config.vocabSize=64
config.gradientReportInterval=1  # Compare every-step reporting fairly.
from types import SimpleNamespace
from babyLLM import BABYLLM
class Counsellor:
 @contextlib.contextmanager
 def infodump(self,*a,**k):yield lambda *a:None
class Style:
 def S_apply(self,*a):return str(a[-1])
 def S_getStat(self,*a):return 'normal'
lib=SimpleNamespace(vocabList=[str(i) for i in range(64)],vocabSize=64,indexToToken={i:str(i) for i in range(64)})
model=BABYLLM(Counsellor(),Style(),None,lib,8,_device=torch.device('mps'))
model.to('mps');model.train();model.nextPixelTarget=None
oldtree=ast.parse(subprocess.check_output(['git','-C',str(ROOT),'show','1fd96ed86036a14d874ed5e67eb18c2464a7b096:babyLLM.py'],text=True))
names = ('_backward_locked', '_print_excitement_report', '_excite_current_grad_norm', '_snapshot_gradients')
ns=dict(sys.modules['babyLLM'].__dict__)
for name in names:
 node=next(n for n in ast.walk(oldtree) if isinstance(n,ast.FunctionDef) and n.name==name)
 exec(compile(ast.Module(body=[node],type_ignores=[]),'old_backward','exec'),ns)
 setattr(model, name, types.MethodType(ns[name],model))
oldbackward=model._backward_locked
newbackward=types.MethodType(BABYLLM._backward_locked,model)
oldreport=model._print_excitement_report
newreport=types.MethodType(BABYLLM._print_excitement_report,model)
samples={'before':[],'after':[]}
prof=cProfile.Profile();prof.enable()
for i in range(10):
 label='before' if i%2==0 else 'after'
 model._backward_locked=oldbackward if label=='before' else newbackward
 model._print_excitement_report=oldreport if label=='before' else newreport
 model.optimizer.zero_grad();t=time.perf_counter()
 logits=model.forward(torch.tensor([1,2,3,4],device='mps'));torch.mps.synchronize();f=time.perf_counter()
 loss=model.computeLoss(logits,5);torch.mps.synchronize();l=time.perf_counter()
 model.backward(loss,0.);torch.mps.synchronize();b=time.perf_counter()
 if i>=2:samples[label].append(b-l)
 print('TIMINGS',i,label,'forward',f-t,'loss',l-f,'backward',b-l,flush=True)
prof.disable();pstats.Stats(prof).sort_stats('cumtime').print_stats(25)

print('STEADY_MEDIANS', {k:statistics.median(v) for k,v in samples.items()})
