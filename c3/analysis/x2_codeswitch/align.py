import json, os, glob, sys
from pyannote.core import Annotation, Segment, Timeline
R = os.path.expanduser("~/mnt/Research")
TOK = f"{R}/Dataset Github/Git/annotations/c1"
TR = f"{R}/C3 Pilot/rttm_truth"
def rttm(p):
    a = Annotation()
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
rows=[]
for tf in sorted(glob.glob(f"{TOK}/*.tokens.jsonl")):
    rid = os.path.basename(tf).replace(".tokens.jsonl","")
    short = rid.replace("J26DS313_","")
    clip = rid if os.path.exists(f"{TR}/{rid}.rttm") else (short if os.path.exists(f"{TR}/{short}.rttm") else None)
    if not clip: rows.append((rid,'-','no truth')); continue
    toks=[]; bad=0
    for l in open(tf,encoding='utf-8'):
        try:
            d=json.loads(l); s,e=float(d['start']),float(d['end']); toks.append((s,e,d['lang'],d.get('switch')))
        except Exception: bad+=1
    ref=rttm(f"{TR}/{clip}.rttm"); u=uem(f"{TR}/{clip}.uem"); sp=ref.get_timeline().support()
    inside=[t for t in toks if u.overlapping((t[0]+t[1])/2)]
    hit=sum(1 for t in inside if sp.overlapping((t[0]+t[1])/2))
    tmax=max(t[1] for t in toks); rmax=sp[-1].end; uend=u.extent().end
    # speaker agreement: C1 utterance speakers vs truth speakers (best mapping by overlap)
    tj=f"{TOK}/{rid}.transcript.json"; agree='-'
    try:
        U=json.load(open(tj,encoding='utf-8'))['utterances']
        from collections import defaultdict
        ov=defaultdict(float); tot=0
        for ut in U:
            seg=Segment(float(ut['start']),float(ut['end']))
            for s2,_,lab in ref.crop(seg).itertracks(yield_label=True):
                ov[(ut['speaker'],lab)]+=s2.duration; tot+=s2.duration
        best={}
        for (a,b),v in sorted(ov.items(),key=lambda kv:-kv[1]):
            if a not in best and b not in best.values(): best[a]=b
        agree=f"{100*sum(ov[(a,b)] for a,b in best.items())/tot:.0f}%" if tot else '0'
    except Exception as ex: agree=f'err {ex}'
    rows.append((rid,clip,f"tok={len(toks)} bad={bad} inUEM={len(inside)} inSpeech={100*hit/max(1,len(inside)):.0f}% tokEnd={tmax:.1f} refEnd={rmax:.1f} uemEnd={uend:.1f} spkAgree={agree} switches={sum(1 for t in toks if t[3])}"))
for r in rows: print(*r, sep=' | ')
