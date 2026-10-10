"""
Live demonstration of C4 for the PP1 panel: a local web page, fully offline.

    python demo.py                 # then open http://127.0.0.1:8765

Type or paste any Sinhala-English text and see:
  * every detected identifier highlighted by type
  * the redacted, shareable text
  * each entity with its placeholder, label and role -- one person keeps
    one colour and one placeholder across Latin and Sinhala script

Binds to 127.0.0.1 only and loads nothing from the internet (NFR1).
"""
from __future__ import annotations

import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from redact import redact_recording

_DETECTOR = None


def detector():
    global _DETECTOR
    if _DETECTOR is None:
        try:
            from ner import HybridDetector, default_model
            _DETECTOR = HybridDetector(default_model())
            _DETECTOR("warm-up")
        except FileNotFoundError:
            from rules import detect
            _DETECTOR = detect
    return _DETECTOR


def kind_of(doc_id: str) -> str:
    if "_summary_" in doc_id:
        return "Summary"
    if doc_id.endswith("_si_v1"):
        return "Clean Sinhala (C2)"
    if "_c2_" in doc_id:
        return "Clean English (C2)"
    return "Transcript (C1)"


def analyse(texts: dict[str, str], rid: str) -> dict:
    started = time.perf_counter()
    result = redact_recording(texts, rid, detect=detector())
    ms = (time.perf_counter() - started) * 1000
    spans: dict[str, list] = {d: [] for d in texts}
    for i, e in enumerate(result.entities):
        for m in e["mentions"]:
            spans[m["doc_id"]].append({"start": m["start"], "end": m["end"], "label": e["label"],
                                       "placeholder": e["placeholder"], "entity": i})
    docs = [{"doc_id": d, "kind": kind_of(d), "original": t, "redacted": result.texts[d],
             "spans": sorted(spans[d], key=lambda s: s["start"])} for d, t in texts.items()]
    scripts = {}
    for i, e in enumerate(result.entities):
        has_si = any(any("඀" <= ch <= "෿" for ch in m["surface"]) for m in e["mentions"])
        has_la = any(not any("඀" <= ch <= "෿" for ch in m["surface"]) for m in e["mentions"])
        scripts[i] = "Latin + Sinhala" if has_si and has_la else "Sinhala" if has_si else "Latin"
    return {
        "docs": docs,
        "entities": [{"placeholder": e["placeholder"], "label": e["label"], "role": e["role"],
                      "mentions": len(e["mentions"]), "scripts": scripts[i]}
                     for i, e in enumerate(result.entities)],
        "leaks": len(result.leaks),
        "ms": round(ms),
    }


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>C4 PII Redaction</title>
<style>
:root{--bg:#f6f7f9;--card:#fff;--ink:#1b2430;--mute:#5d6b7a;--line:#dfe3e8;--accent:#1f5fae}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.55 "Segoe UI","Iskoola Pota","Nirmala UI",system-ui,sans-serif}
header{background:#13294b;color:#fff;padding:18px 28px}
header h1{margin:0;font-size:20px}header p{margin:4px 0 0;color:#c6d3e6;font-size:13px}
main{max-width:1280px;margin:0 auto;padding:20px 28px 40px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px 18px;margin-bottom:16px}
textarea{width:100%;min-height:96px;border:1px solid var(--line);border-radius:8px;padding:10px;font:inherit;resize:vertical}
.row{display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin-top:10px}
button,select{font:inherit;padding:8px 14px;border-radius:8px;border:1px solid var(--line);background:#fff;cursor:pointer}
button:disabled{opacity:.55;cursor:wait}
button.primary{background:var(--accent);color:#fff;border-color:var(--accent)}
.stats{display:flex;gap:22px;flex-wrap:wrap;color:var(--mute);font-size:13px}
.stats b{color:var(--ink);font-size:18px;display:block}
.doc{display:grid;grid-template-columns:1fr 1fr;gap:14px;padding:10px 0;border-top:1px solid var(--line)}
.doc:first-of-type{border-top:0}.kind{grid-column:1/-1;font-size:12px;color:var(--mute);text-transform:uppercase;letter-spacing:.04em}
.pane{white-space:pre-wrap;word-wrap:break-word}
.tag{border-radius:4px;padding:1px 3px;box-shadow:inset 0 -2px 0 rgba(0,0,0,.18)}
.tag sub{font-size:10px;font-weight:600;margin-left:3px;opacity:.75}
.ph{background:#e8eef7;border-radius:4px;padding:1px 4px;font-weight:600;color:#13294b}
table{width:100%;border-collapse:collapse;font-size:14px}th,td{text-align:left;padding:7px 8px;border-bottom:1px solid var(--line)}
th{color:var(--mute);font-weight:600;font-size:12px;text-transform:uppercase}
.sw{display:inline-block;width:12px;height:12px;border-radius:3px;vertical-align:-1px;margin-right:6px}
.cols{display:grid;grid-template-columns:1fr 1fr;gap:14px;font-size:12px;color:var(--mute);text-transform:uppercase}
.legend span{margin-right:12px;font-size:13px}
.ok{color:#1e7a3c}.bad{color:#b3261e}
@media(max-width:800px){.doc,.cols{grid-template-columns:1fr}}
</style></head><body>
<header><h1>C4 — Offline PII Detection and Redaction for Sinhala-English text</h1>
<p>J26-DS-313 · Component 4 · runs entirely on this laptop, no network</p></header>
<main>
<div class="card">
  <textarea id="text" placeholder="Paste Singlish / Sinhala / English text here...">hello, mama Amal, Seylan Bank eken kathaa karanne. Mage nama Nimal Perera, NIC eka 953201456V, mobile number eka binduwai hatai hatai ekai dekai thunai hatharai pahai hayai hatha. මගේ නම නිමල් පෙරේරා, ලිපිනය 24, ගාලු පාර, දෙහිවල.</textarea>
  <div class="row">
    <button class="primary" id="b1" onclick="runText()">Redact this text</button>
    <span id="status" style="color:var(--mute)"></span>
  </div>
</div>
<div class="card" id="summary" hidden>
  <h2 id="heading" style="margin:0 0 10px;font-size:17px"></h2>
  <div class="stats" id="stats"></div>
  <div class="legend" id="legend" style="margin-top:10px"></div>
</div>
<div class="card" id="entities" hidden><table><thead><tr><th>Placeholder</th><th>Type</th><th>Role</th><th>Scripts</th><th>Mentions</th></tr></thead><tbody id="ebody"></tbody></table></div>
<div class="card" id="docs" hidden><div class="cols"><div>Original — detected identifiers</div><div>Redacted — shareable output</div></div><div id="dbody"></div></div>
</main>
<script>
const COLORS=["#ffd6a5","#caffbf","#9bf6ff","#bdb2ff","#ffc6ff","#fdffb6","#a0c4ff","#ffadad","#d0f4de","#e4c1f9","#fcf6bd","#b9fbc0"];
const LABEL={PERSON:"Person",NIC:"NIC",PHONE:"Phone",ADDRESS:"Address",ACCOUNT:"Account",DOB:"Date of birth",EMAIL:"Email",ORG:"Organisation",LOCATION:"Location"};
const esc=s=>s.replace(/[&<>]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));
function highlight(t,spans){let o="",c=0;for(const s of spans){if(s.start<c)continue;
 o+=esc(t.slice(c,s.start))+`<span class="tag" style="background:${COLORS[s.entity%COLORS.length]}" title="${s.placeholder}">${esc(t.slice(s.start,s.end))}<sub>${s.label}</sub></span>`;c=s.end}
 return o+esc(t.slice(c))}
function phs(t){return esc(t).replace(/\[[A-Z]+_\d+\]/g,m=>`<span class="ph">${m}</span>`)}
async function post(body,title){
 const st=document.getElementById("status"),btns=[document.getElementById("b1")];
 btns.forEach(b=>b.disabled=true);st.className="";st.textContent="Working…";
 try{const r=await fetch("/api/redact",{method:"POST",body:JSON.stringify(body)});
  if(!r.ok)throw new Error(await r.text());
  const d=await r.json();d.title=title;show(d);st.textContent="Done.";
  document.getElementById("summary").scrollIntoView({behavior:"smooth"})}
 catch(e){st.className="bad";st.textContent="Error: "+e.message}
 finally{btns.forEach(b=>b.disabled=false)}}
function runText(){post({text:document.getElementById("text").value},"Your text")}
function show(d){
 document.getElementById("heading").textContent=`${d.title} — ${d.docs.length} document${d.docs.length>1?"s":""}`;
 const people=d.entities.filter(e=>e.label==="PERSON"),linked=people.filter(e=>e.scripts==="Latin + Sinhala").length;
 document.getElementById("stats").innerHTML=`<div><b>${d.entities.length}</b>entities redacted</div><div><b>${people.length}</b>people</div><div><b>${linked}</b>linked across scripts</div><div><b>${d.docs.length}</b>documents</div><div><b>${d.ms} ms</b>processing</div><div><b class="${d.leaks?"bad":"ok"}">${d.leaks?d.leaks+" leak(s)":"0 leaks"}</b>leak check</div>`;
 document.getElementById("legend").innerHTML="Same colour = same entity, in every document and script.";
 document.getElementById("ebody").innerHTML=d.entities.map((e,i)=>`<tr><td><span class="sw" style="background:${COLORS[i%COLORS.length]}"></span><span class="ph">${e.placeholder}</span></td><td>${LABEL[e.label]||e.label}</td><td>${e.role||"—"}</td><td>${e.scripts}</td><td>${e.mentions}</td></tr>`).join("");
 document.getElementById("dbody").innerHTML=d.docs.map(x=>`<div class="doc"><div class="kind">${x.kind}</div><div class="pane">${highlight(x.original,x.spans)}</div><div class="pane">${phs(x.redacted)}</div></div>`).join("");
 for(const id of["summary","entities","docs"])document.getElementById(id).hidden=false}

</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/":
            self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):
        if self.path != "/api/redact":
            return self._send(404, b"not found", "text/plain")
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        # Only the text typed on the page is processed; no stored data is loaded.
        rid, texts = "input", {"input_transcript_u001_v1": body.get("text", "")}
        try:
            result = analyse(texts, rid)
        except Exception as exc:                      # shown on the page, not swallowed
            return self._send(500, f"{type(exc).__name__}: {exc}".encode("utf-8"),
                              "text/plain; charset=utf-8")
        self._send(200, json.dumps(result, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8")

    def log_message(self, *args):
        pass


def main() -> None:
    ap = argparse.ArgumentParser(description="C4 live demo")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    print("loading models (about a minute) ...")
    analyse({"warm_transcript_u001_v1": "mage nama Nimal Perera, NIC eka 953201456V."}, "warm-up")
    print(f"C4 demo running offline at http://127.0.0.1:{args.port}  (Ctrl+C to stop)")
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
