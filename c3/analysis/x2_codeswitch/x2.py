import json, os, glob, random
import numpy as np
from pyannote.core import Annotation, Segment, Timeline
from pyannote.metrics.diarization import DiarizationErrorRate
import sys
R=os.path.expanduser("~/mnt/Research"); TOK=f"{R}/Dataset Github/Git/annotations/c1"
TR=f"{R}/C3 Pilot/rttm_truth"; HY=f"{R}/C3 Pilot/rttm_draft"; STEP=0.01; W=float(sys.argv[1]) if len(sys.argv)>1 else 1.0
def rttm(p,uri):
    a=Annotation(uri=uri)
    for i,l in enumerate(open(p,encoding='utf-8')):
        x=l.split()
        if len(x)>=8 and x[0]=='SPEAKER' and float(x[4])>0: a[Segment(float(x[3]),float(x[3])+float(x[4])),i]=x[7]
    return a
def uem(p):
    t=Timeline()
    for l in open(p,encoding='utf-8'):
        x=l.split()
        if len(x)>=4: t.add(Segment(float(x[2]),float(x[3])))
    return t
def frames(ann,labels,n):
    M=np.zeros((len(labels),n),bool); idx={l:i for i,l in enumerate(labels)}
    for s,_,l in ann.itertracks(yield_label=True):
        M[idx[l],int(s.start/STEP):min(n,int(s.end/STEP))]=True
    return M
res={}
for tf in sorted(glob.glob(f"{TOK}/*.tokens.jsonl")):
    rid=os.path.basename(tf)[:-13]; short=rid.replace("J26DS313_","")
    clip=rid if os.path.exists(f"{TR}/{rid}.rttm") else (short if os.path.exists(f"{TR}/{short}.rttm") else None)
    if not clip or clip=="R0052" or not os.path.exists(f"{HY}/{clip}.rttm"): continue
    toks=[json.loads(l) for l in open(tf,encoding='utf-8')]
    ref=rttm(f"{TR}/{clip}.rttm",clip); hyp=rttm(f"{HY}/{clip}.rttm",clip); u=uem(f"{TR}/{clip}.uem")
    mp=DiarizationErrorRate().optimal_mapping(ref,hyp.crop(u))
    hyp=hyp.rename_labels(mapping={h:mp.get(h,f"X_{h}") for h in hyp.labels()})
    end=max(u.extent().end, ref.get_timeline().extent().end, hyp.get_timeline().extent().end)+1
    n=int(end/STEP)+1
    rl=ref.labels(); hl=hyp.labels()
    RM=frames(ref,rl,n); HM=frames(hyp,hl,n)
    valid=np.zeros(n,bool)
    for s in u: valid[int(s.start/STEP):int(s.end/STEP)]=True
    t0=min(t['start'] for t in toks); t1=max(t['end'] for t in toks)
    span=np.zeros(n,bool); span[int(t0/STEP):int(t1/STEP)]=True; valid&=span
    nref=RM.sum(0); single=(nref==1)&valid; ovl=(nref>=2)&valid
    refspk=np.where(single, RM.argmax(0), -1)
    # hyp label for each frame: single hyp label mapped to ref name; multiple/none handled
    hyp_name=np.array([None]*n,dtype=object)
    hcount=HM.sum(0)
    for j,l in enumerate(hl):
        hyp_name[HM[j]&(hcount==1)]=l
    conf=np.zeros(n,bool)
    for i in np.where(single)[0]:
        if hcount[i]>=1:
            names={hl[j] for j in range(len(hl)) if HM[j,i]}
            if rl[refspk[i]] not in names: conf[i]=True
    # switch points
    sw=[]
    prev=None
    for t in toks:
        if prev and prev['utt_id']==t['utt_id'] and {prev['lang'],t['lang']}=={'SI','EN'}:
            sw.append((prev['end']+t['start'])/2)
        prev=t
    near=np.zeros(n,bool)
    for x in sw: near[max(0,int((x-W)/STEP)):int((x+W)/STEP)]=True
    # control: ref speaker changes
    chg=np.zeros(n,bool)
    segs=sorted(ref.itertracks(yield_label=True),key=lambda z:z[0].start)
    for (a,_,la),(b,_,lb) in zip(segs,segs[1:]):
        if la!=lb:
            for x in (a.end,b.start): chg[max(0,int((x-W)/STEP)):int((x+W)/STEP)]=True
    base=single&~chg
    N=base&near; F=base&~near
    # M2: spurious hyp changes inside single-ref stretches (frame-to-frame change of single hyp name)
    def changes(mask):
        c=0
        for i in np.where(mask[1:]&mask[:-1])[0]:
            a,b=hyp_name[i],hyp_name[i+1]
            if a is not None and b is not None and a!=b: c+=1
        return c
    res[clip]=dict(nsw=len(sw), near_s=N.sum()*STEP, far_s=F.sum()*STEP,
        near_conf=conf[N].sum()*STEP, far_conf=conf[F].sum()*STEP,
        near_chg=changes(N), far_chg=changes(F),
        ovl_s=ovl.sum()*STEP, ovl_conf=0.0)
tot=lambda k: sum(r[k] for r in res.values())
nr=tot('near_conf')/tot('near_s'); fr=tot('far_conf')/tot('far_s')
print(f"clips={len(res)} switches={tot('nsw')} near={tot('near_s')/60:.1f} min far={tot('far_s')/60:.1f} min")
print(f"M1 confusion near={100*nr:.2f}% far={100*fr:.2f}% ratio={nr/fr if fr else float('inf'):.2f}")
m2n=tot('near_chg')/(tot('near_s')/60); m2f=tot('far_chg')/(tot('far_s')/60)
print(f"M2 spurious changes/min near={m2n:.2f} far={m2f:.2f} ratio={m2n/m2f if m2f else float('inf'):.2f}")
keys=list(res); random.seed(0); diffs=[]
for _ in range(10000):
    s=[random.choice(keys) for _ in keys]
    ns=sum(res[k]['near_s'] for k in s); fs=sum(res[k]['far_s'] for k in s)
    if ns and fs: diffs.append(sum(res[k]['near_conf'] for k in s)/ns - sum(res[k]['far_conf'] for k in s)/fs)
lo,hi=np.percentile(diffs,[2.5,97.5]); print(f"bootstrap 95% CI (near-far) M1: {100*lo:.2f} to {100*hi:.2f} pp")
both=[k for k in keys if res[k]['near_s']>5 and res[k]['far_s']>5]
up=sum(1 for k in both if res[k]['near_conf']/res[k]['near_s']>res[k]['far_conf']/res[k]['far_s'])
print(f"clips near>far: {up}/{len(both)}")
print("clip | switches | near s | far s | conf near % | conf far %")
for k,r in res.items():
    print(f"{k} | {r['nsw']} | {r['near_s']:.0f} | {r['far_s']:.0f} | {100*r['near_conf']/max(r['near_s'],1e-9):.1f} | {100*r['far_conf']/max(r['far_s'],1e-9):.1f}")
json.dump(res,open(os.path.expanduser("~/x2/x2_res.json"),"w"),indent=1,default=float)
