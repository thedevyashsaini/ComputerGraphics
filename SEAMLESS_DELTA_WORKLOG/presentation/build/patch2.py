import os,re
H=r"C:\Users\Devyash\Documents\Computer_Graphics\SEAMLESS_DELTA_WORKLOG\presentation\html"
p=os.path.join(H,"slide23.html"); t=open(p,encoding="utf-8").read()
t=t.replace('max-height:360px','max-height:290px').replace('<table class="t" style="font-size:24px">','<table class="t" style="font-size:23px">')
t=t.replace('<div class="glass tcard" style="flex:1.35">','<div class="glass tcard" style="flex:1.35;padding:10px 18px">')
t=t.replace('<td>','<td style="padding:9px 14px">')
open(p,"w",encoding="utf-8").write(t)
p=os.path.join(H,"slide24.html"); t=open(p,encoding="utf-8").read()
t=t.replace('grid-template-columns:1fr 1fr;gap:22px;align-content:start','grid-template-columns:1fr 1fr;grid-auto-rows:1fr;gap:24px;align-content:stretch')
t=t.replace('<div class="glass" style="padding:22px 26px">','<div class="glass" style="padding:26px 34px;display:flex;flex-direction:column;justify-content:center">')
t=t.replace('font-size:29px;font-weight:700;color:#12301b','font-size:36px;font-weight:800;color:#0b2214').replace('class="note" style="font-size:23px;margin-top:6px"','class="note" style="font-size:27px;margin-top:8px"')
open(p,"w",encoding="utf-8").write(t)