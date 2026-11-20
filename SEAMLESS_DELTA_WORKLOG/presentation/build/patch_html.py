import re, os, glob
H=r"C:\Users\Devyash\Documents\Computer_Graphics\SEAMLESS_DELTA_WORKLOG\presentation\html"
def P(n): return os.path.join(H, f"slide{n:02d}.html")
def R(n,a,b,cnt=1):
    t=open(P(n),encoding="utf-8").read()
    if a not in t: print("MISSING", n, a[:60]); return
    t=t.replace(a,b); open(P(n),"w",encoding="utf-8").write(t)
# kicker on all content slides
for n in range(2,27):
    t=open(P(n),encoding="utf-8").read()
    sec=re.search(r'<span class="sec">([^<]+)</span>',t).group(1)
    if 'class="kicker"' not in t:
        t=t.replace('<div class="frame"><h1>', f'<div class="frame"><div class="kicker"><i></i>{sec}&nbsp;&nbsp;{n:02d}</div><h1>',1)
    open(P(n),"w",encoding="utf-8").write(t)
R(1,'<p class="base">','<p class="base glass">')
R(6,'rgba(232,242,86,.65)','rgba(47,168,74,.30)')
R(6,'rgba(140,224,112,.55)','rgba(150,226,160,.75)')
R(12,'<div class="math" style="font-size:31px">w<sub>i</sub>','<div class="math" style="font-size:27px">w<sub>i</sub>')
R(12,'<div class="math" style="font-size:31px">S(t)','<div class="math" style="font-size:30px">S(t)')
R(13,'<div class="math" style="font-size:31px">L = L','<div class="math" style="font-size:28px">L = L')
t=open(P(11),encoding="utf-8").read()
t=t.replace('<div class="glass stat" style="gap:18px">','<div class="glass stat" style="gap:26px;justify-content:center;padding:44px 40px">')
t=t.replace('<div style="font-size:34px;font-weight:700;color:#12301b">','<div style="font-size:42px;font-weight:800;line-height:1.15;color:#0b2214">')
t=t.replace('<div class="lab">','<div class="lab" style="font-size:29px">')
open(P(11),"w",encoding="utf-8").write(t)
# slide 23 layout
t=open(P(23),encoding="utf-8").read()
t=re.sub(r'<div class="glass fig" style="flex:1.25;justify-content:center">(<img[^>]*>)</div>\s*<div class="col" style="flex:1"><div class="glass tcard"><table class="t" style="font-size:25px">',
 lambda m:'<div class="stack"><div class="glass fig" style="flex:none">'+m.group(1).replace('style=""','style="max-height:360px"')+'</div><div class="row" style="flex:1;align-items:stretch"><div class="glass tcard" style="flex:1.35"><table class="t" style="font-size:24px">',t)
t=re.sub(r'</table></div>\s*<div class="note">(About 22%[^<]*)</div></div>',
 '</table></div><div class="glass tint mathbox" style="flex:1;display:flex;flex-direction:column;justify-content:center"><div class="label">BOTTOM LINE</div><div class="note">About 22% slower point queries, since overlap points evaluate 2 to 8 bricks. Frame time shows no consistent penalty. Blending itself adds no parameters and no training.</div></div></div></div>',t)
open(P(23),"w",encoding="utf-8").write(t)
# slide 25 add claim-strength table
t=open(P(25),encoding="utf-8").read()
if "Claim</th>" not in t:
    t=t.replace('</ul></div></div></div><div class="foot">','''</ul></div>
<div class="col" style="flex:.85"><div class="glass tcard"><table class="t" style="font-size:26px">
 <tr><th>Claim</th><th>Seeds</th><th>Strength</th></tr>
 <tr class="ours"><td>Ours beats g16 hard</td><td>3</td><td>every seed</td></tr>
 <tr><td>Blend beats g4 hard</td><td>3</td><td>every seed</td></tr>
 <tr><td>Recon FT beats consistency</td><td>2</td><td>indicative</td></tr>
 <tr><td>g16 needs 2.3&times; params</td><td>1</td><td>indicative</td></tr>
</table></div></div></div></div><div class="foot">''')
open(P(25),"w",encoding="utf-8").write(t)
print("23 stack:", "BOTTOM LINE" in open(P(23),encoding="utf-8").read(), " 25 table:", "Claim</th>" in open(P(25),encoding="utf-8").read())