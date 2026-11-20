p=r"C:\Users\Devyash\Documents\Computer_Graphics\SEAMLESS_DELTA_WORKLOG\presentation\build\gen_slides.py"
t=open(p,encoding="utf-8").read()
def R(a,b):
    global t
    if a not in t: print("MISSING:", a[:70]); return
    t=t.replace(a,b)
R('''inner = f'<div class="frame"><h1>{s["title"]}</h1>''','''inner = f'<div class="frame"><div class="kicker"><i></i>{s["sec"]}&nbsp;&nbsp;{n:02d}</div><h1>{s["title"]}</h1>''')
R('<p class="base">Base paper','<p class="base glass">Base paper')
R('rgba(232,242,86,.65)','rgba(47,168,74,.30)')
R('rgba(140,224,112,.55)','rgba(150,226,160,.75)')
R("""<div class="glass fig" style="flex:1.25;justify-content:center">''' + img("tradeoffs.png") + '''</div>
<div class="col" style="flex:1"><div class="glass tcard"><table class="t" style="font-size:25px">""",
"""<div class="stack"><div class="glass fig" style="flex:none">''' + img("tradeoffs.png", "max-height:360px") + '''</div>
<div class="row" style="flex:1;align-items:stretch"><div class="glass tcard" style="flex:1.35"><table class="t" style="font-size:24px">""")
R("""</table></div>
<div class="note">About 22% slower queries, since overlap points evaluate 2 to 8 bricks. Frame time shows no consistent penalty.</div></div>''',""",
"""</table></div>
<div class="glass tint mathbox" style="flex:1;display:flex;flex-direction:column;justify-content:center"><div class="label">BOTTOM LINE</div><div class="note">About 22% slower point queries, since overlap points evaluate 2 to 8 bricks. Frame time shows no consistent penalty. Blending itself adds no parameters and no training.</div></div></div></div>''',""")
R('<div class="math" style="font-size:31px">w<sub>i</sub>','<div class="math" style="font-size:27px">w<sub>i</sub>')
R('<div class="math" style="font-size:31px">S(t)','<div class="math" style="font-size:30px">S(t)')
R('<div class="math" style="font-size:31px">L = L','<div class="math" style="font-size:28px">L = L')
R("""<div class="glass stat" style="gap:18px"><span class="num">{n}</span><div style="font-size:34px;font-weight:700;color:#12301b">{t}</div><div class="lab">{d}</div></div>""",
  """<div class="glass stat" style="gap:26px;justify-content:center;padding:44px 40px"><span class="num">{n}</span><div style="font-size:42px;font-weight:800;line-height:1.15;color:#0b2214">{t}</div><div class="lab" style="font-size:29px">{d}</div></div>""")
R("""], "pts sm") + '''</div>''',
"\"\"\"- One dataset""", "")
i=t.index('add("Extensions", "Limitations"')
j=t.index("</div>''',", i)
t=t[:j]+"""</div>
<div class="col" style="flex:.85"><div class="glass tcard"><table class="t" style="font-size:26px">
 <tr><th>Claim</th><th>Seeds</th><th>Strength</th></tr>
 <tr class="ours"><td>Ours beats g16 hard</td><td>3</td><td>every seed</td></tr>
 <tr><td>Blend beats g4 hard</td><td>3</td><td>every seed</td></tr>
 <tr><td>Recon FT beats consistency</td><td>2</td><td>indicative</td></tr>
 <tr><td>g16 needs 2.3&times; params</td><td>1</td><td>indicative</td></tr>
</table></div>"""+t[j:]
open(p,"w",encoding="utf-8").write(t)
print("ok")