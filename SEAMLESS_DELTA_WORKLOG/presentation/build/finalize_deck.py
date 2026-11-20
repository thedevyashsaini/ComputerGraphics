from pathlib import Path
import json, re, shutil

ROOT = Path(__file__).resolve().parents[1]
HTML = ROOT / "html"
SRC = ROOT / "build" / "source_26"
SRC.mkdir(exist_ok=True)

for i in range(1, 27):
    source = HTML / f"slide{i:02d}.html"
    backup = SRC / source.name
    if not backup.exists():
        shutil.copy2(source, backup)

slides = [(SRC / f"slide{i:02d}.html").read_text(encoding="utf-8") for i in range(1, 27)]
notes = json.loads((ROOT / "build" / "notes.json").read_text(encoding="utf-8"))[:26]

head = '<!doctype html><html><head><meta charset="utf-8"><link rel="stylesheet" href="style.css"></head><body><div class="slide"><div class="grid-tex"></div><div class="blob b1"></div><div class="blob b2"></div><div class="blob b3"></div><div class="blob b4"></div>'

def shell(section, title, body):
    return f'''{head}<div class="frame"><div class="kicker"><i></i>{section}&nbsp;&nbsp;00</div><h1>{title}</h1><div class="body">{body}</div></div><div class="foot"><span class="sec">{section}</span><span>Seamless APMGSRN</span><span>00 / 31</span></div></div></body></html>'''

slides[0] = f'''{head}
<div class="cover-grid">
 <div class="cover-copy">
  <span class="pill">Computer Graphics and Multimedia, Course Project</span>
  <h1 class="hero">Seamless<br><span class="hl">APMGSRN</span></h1>
  <p class="sub">Smooth interfaces for brick-decomposed neural volumes using four ghost cells</p>
  <div class="teamline"><span><b>Devyash Saini</b><small>202351030</small></span><span><b>Deepanshu</b><small>202351025</small></span><span><b>Himanshu</b><small>202351047</small></span></div>
  <p class="base">Based on Wurster et al., <i>Adaptively Placed Multi-Grid Scene Representation Networks for Large-Scale Data Visualization</i>, IEEE TVCG 2024</p>
 </div>
 <div class="cover-visual glass"><img src="../assets/gt.png"><div class="cover-tag"><b>Plume</b><span>512 × 128 × 128</span></div></div>
</div></div></body></html>'''

slides[2] = shell("Preliminaries", "Implicit neural representations", '''
<div class="col" style="flex:1.2"><ul class="pts"><li>A neural network f<sub>&theta;</sub>(x, y, z) learns to reproduce the scalar value at any coordinate</li><li>The weights become the compressed file, so storage depends on model size and not on resolution</li><li>Random access comes for free: a renderer can query any point without decompressing the volume</li><li>Training uses random point batches with an L2 loss against the trilinear ground truth</li></ul></div>
<div class="col" style="flex:1"><div class="glass tint mathbox"><div class="label">OBJECTIVE</div><div class="math">min<sub>&theta;</sub> &nbsp;E<sub>x</sub> [ (f<sub>&theta;</sub>(x) &minus; v(x))<sup>2</sup> ]</div></div>
<div class="glass compression-strip"><div><strong>32 MiB</strong><span>raw float32 volume</span></div><div class="compress-mark">9× smaller</div><div><strong>3.5 MiB</strong><span>923k-parameter model</span></div></div></div>''')

slides[3] = shell("Preliminaries", "Feature-grid encoders", '''
<div class="col" style="flex:1"><ul class="pts sm"><li>Pure MLPs learn fine detail slowly, so modern methods store learnable features in a grid</li><li>A query trilinearly interpolates grid features and a tiny MLP decodes them to a scalar</li><li>fVSRN uses one dense grid, NGP uses multiresolution hash grids</li><li>APMGSRN uses several small grids that move to where the data needs detail</li></ul></div>
<div class="glass fig fit-card" style="flex:1.55"><img src="../assets/architecture_comparison.jpg"><div class="caption">Encoder comparison from Wurster et al. 2024</div></div>''')

slides[4] = shell("Preliminaries", "APMGSRN: adaptively placed grids", '''
<div class="stack"><div class="glass fig architecture-card"><img src="../assets/architecture.jpg"></div><div class="row" style="flex:1;min-height:0">
<div class="col" style="flex:1.05"><ul class="pts sm"><li>M small feature grids, each with its own learned affine transform</li><li>A KL loss moves grid density toward regions with high error</li><li>Grid movement stops by 80% of training, then the grids refine in place</li></ul></div>
<div class="glass tint mathbox density-card" style="flex:1"><div class="label">FLAT-TOP GAUSSIAN DENSITY</div><div class="math" style="font-size:31px">&rho;<sub>i</sub>(x) &prop; det(G<sub>i</sub>) &middot; exp(&minus;&frac12; &Sigma; G<sub>i</sub>(x)<sup>20</sup>)</div><div class="density-points"><span><b>2 optimizers</b> grids and decoder</span><span><b>2 × 64</b> decoder MLP</span></div></div></div></div>''')

slides[5] = shell("Preliminaries", 'Domain decomposition and <span class="hl">ghost cells</span>', '''
<div class="col" style="flex:1"><ul class="pts sm"><li>Large volumes do not fit one model or one GPU, so the paper splits the domain into n × n × n bricks</li><li>One independent APMGSRN per brick, trained in parallel</li><li>Ghost cells extend each brick into its neighbours so both models see the shared boundary</li><li>Hard router: brick index = floor((x + 1) / 2 × n)</li></ul></div>
<div class="glass overlap-card" style="flex:1.1"><div class="label">1D VIEW OF TWO BRICKS ALONG X</div><div class="overlap-diagram"><div class="oa">Brick A model</div><div class="ob">Brick B model</div><div class="zone">SHARED GHOST-CELL REGION</div><div class="router"></div><div class="router-label">hard router split</div></div><div class="note">Both models cover the overlap. The hard router still selects only one prediction at each point.</div></div>''')

slides[7] = shell("Problem", "Seams at brick boundaries", '''
<div class="glass fig fit-card paper-seam-card" style="flex:1.45"><img src="../assets/paper_ghostcells.jpg"><div class="caption">Wurster et al. 2024: 1 ghost cell shows a seam, 16 ghost cells hide it</div></div><div class="col" style="flex:1"><ul class="pts sm"><li>Two independently trained bricks disagree slightly at the split, so the field jumps</li><li>Isosurfaces and shading show the jump as a crack or a bright line</li><li>The paper widens the overlap to 16 ghost cells and accepts about 1 dB less PSNR</li><li>Their future work lists a communication scheme between bricks</li></ul></div>''')

slides[8] = shell("Problem", "Problem statement", '''
<div class="bento"><div class="glass tint goal-tile"><div class="label">TARGET</div><div>Match the seam quality of 16 ghost cells using only <span class="hl">2 to 4 ghost cells</span></div></div>
<div class="glass bento-tile"><div class="label">QUALITY</div><b>Near-zero value jump</b><span>and higher boundary PSNR</span></div>
<div class="glass bento-tile"><div class="label">FAIRNESS</div><b>Same model size</b><span>with the original router retained</span></div>
<div class="glass bento-tile"><div class="label">HARDWARE</div><b>RTX 3050, 4 GB</b><span>about two GPU-hours</span></div>
<div class="glass bento-tile"><div class="label">SCOPE</div><b>Plume</b><span>2 × 2 × 2 decomposition</span></div></div>''')

slides[9] = shell("Implementation", "Reproducing the base paper", '''
<div class="col" style="flex:1"><ul class="pts"><li>Cloned the authors' repository and trained APMGSRN locally on Plume</li><li>Adapted the environment for Windows and a 4 GB GPU</li><li>nerfacc and tcnn could not be built, so we wrote a PyTorch ray marcher</li><li>Verified the paper's findings on ghost cells and grid migration</li></ul></div><div class="glass fig fit-card teaser-card" style="flex:1.2"><img src="../assets/teaser.jpg"><div class="caption">Adaptive grids migrate toward detailed regions during training</div></div>''')

slides[13] = shell("Implementation", "Code and evaluation pipeline", '''
<div class="col" style="flex:1"><ul class="pts sm"><li>Blended query wraps trained per-brick models and plugs into evaluation and rendering</li><li>Hard router kept bit-for-bit identical, checked by a unit test</li><li>19 CPU unit tests cover weights, continuity, corners and router equivalence</li><li>One script per stage with results collected from JSON</li></ul></div><div class="glass fig fit-card plot-card" style="flex:1.05"><img src="../assets/slice_error_g4_hard.png"><div class="caption">g4 hard routing: slice error concentrates at brick splits</div></div>''')

slides[15] = shell("Results", "Headline result against the paper setting", '''
<div class="stack"><div class="row compact-stats"><div class="glass stat"><div class="big">+2.00 dB</div><div class="lab">global PSNR</div></div><div class="glass stat"><div class="big">+2.51 dB</div><div class="lab">boundary PSNR</div></div><div class="glass stat"><div class="big">579×</div><div class="lab">lower value jump</div></div><div class="glass stat"><div class="big">0</div><div class="lab">extra parameters</div></div></div><div class="glass fig headline-card" style="flex:1"><img src="../assets/headline.png"></div></div>''')

slides[17] = shell("Results", "Paired comparisons across seeds", '''
<div class="col" style="flex:1"><div class="glass tcard"><table class="t tight-table"><tr><th>Comparison</th><th>Global<br>dB</th><th>Boundary<br>dB</th><th>Render<br>dB</th><th>Value<br>jump</th><th>Gradient<br>jump</th><th>Wins</th></tr>
<tr class="ours"><td>Ours vs g16 hard</td><td>+2.00</td><td>+2.51</td><td>+2.08</td><td>579×</td><td>1.38×</td><td>3 / 3</td></tr>
<tr><td>g4 blend vs g16 hard</td><td>+1.69</td><td>+2.32</td><td>+1.16</td><td>567×</td><td>1.36×</td><td>3 / 3*</td></tr>
<tr><td>Ours vs g4 hard</td><td>+0.51</td><td>+1.44</td><td>+1.10</td><td>474×</td><td>1.18×</td><td>3 / 3</td></tr>
<tr><td>Blend vs g4 hard</td><td>+0.20</td><td>+1.24</td><td>+0.19</td><td>463×</td><td>1.16×</td><td>3 / 3</td></tr>
<tr><td>Ours vs g16 + recon FT</td><td>+1.50</td><td>+1.77</td><td>+1.08</td><td>445×</td><td>1.32×</td><td>1 / 1</td></tr>
<tr><td>Recon FT + blend vs g16 + recon FT</td><td>+1.73</td><td>+2.17</td><td>+1.36</td><td>442×</td><td>1.31×</td><td>1 / 1</td></tr>
<tr><td>Ours vs g16 32<sup>3</sup></td><td>&minus;0.09</td><td>+0.40</td><td>&minus;0.41</td><td>444×</td><td>1.36×</td><td>1 / 1</td></tr>
<tr><td>Consistency FT vs recon FT</td><td>&minus;0.21</td><td>&minus;0.38</td><td>&minus;0.25</td><td>1.01×</td><td>1.01×</td><td>0 / 2</td></tr></table></div><div class="note small-note">Paired, same-seed differences. “Wins” covers all listed metrics. *Render PSNR: 2 / 3 seeds.</div></div>''')

slides[19] = shell("Results", "How much capacity does g16 need to catch up", '''
<div class="stack"><div class="glass fig capacity-card" style="flex:1"><img src="../assets/capacity_curve.png"></div><div class="row compact-stats"><div class="glass stat"><div class="big">2.3×</div><div class="lab">parameters to tie global PSNR</div></div><div class="glass stat"><div class="big">+0.40 dB</div><div class="lab">our boundary lead remains</div></div><div class="glass stat"><div class="big">444×</div><div class="lab">smaller value jump</div></div></div></div>''')

slides[22] = shell("Results", "Costs and tradeoffs", '''
<div class="stack"><div class="glass fig tradeoff-card"><img src="../assets/tradeoffs.png"></div><div class="row tradeoff-bottom"><div class="glass tcard" style="flex:1.45"><table class="t trade-table"><tr><th>Cost</th><th>Hard</th><th>Blend</th></tr><tr><td>Query rate</td><td>11.5 to 12.2 Mpts/s</td><td>8.7 to 9.6 Mpts/s</td></tr><tr><td>Frame time</td><td>3.5 to 4.3 s</td><td>same range</td></tr><tr><td>Peak GPU memory</td><td>0.181 GB</td><td>0.182 GB</td></tr><tr><td>Parameters</td><td>923k</td><td>923k</td></tr><tr><td>Extra training</td><td>0 min</td><td>4 to 7 min</td></tr></table></div><div class="glass tint bottom-line"><div class="label">BOTTOM LINE</div><b>22% slower point queries</b><span>No consistent frame-time penalty. Blending adds no parameters.</span></div></div></div>''')

full_specs = {
    14: ("Implementation", "Full view: boundary slice error", "slice_error_g4_hard.png", "The error map makes the brick boundaries visible as straight lines."),
    16: ("Results", "Full view: headline comparison", "headline.png", "Use this view when discussing the four headline metrics."),
    20: ("Results", "Full view: capacity comparison", "capacity_curve.png", "The paper setting needs 2.3 times more parameters to match global PSNR."),
    21: ("Results", "Full view: overlap-width comparison", "ghost_cell_curve.png", "Wide hard-routed overlap lowers PSNR while the seam remains."),
    23: ("Results", "Full view: performance tradeoffs", "tradeoffs.png", "The three panels compare throughput, frame time and GPU memory."),
}

def full_slide(section, title, img):
    return shell(section, title, f'<div class="glass full-view"><img src="../assets/{img}"></div>')

new_slides, new_notes = [], []
for idx, (slide, note) in enumerate(zip(slides, notes), start=1):
    new_slides.append(slide)
    if idx == 10:
        note = "Keep this short, about one minute. This establishes that the base implementation and qualitative ghost-cell behaviour were reproduced locally. Teaser figure from the base paper."
    new_notes.append(note)
    if idx in full_specs:
        section, title, img, extra_note = full_specs[idx]
        new_slides.append(full_slide(section, title, img))
        new_notes.append(extra_note)

total = len(new_slides)
assert total == 31
for pos, html in enumerate(new_slides, start=1):
    html = re.sub(r'(<div class="kicker"><i></i>[^<]*?&nbsp;&nbsp;)\d+', rf'\g<1>{pos:02d}', html)
    html = re.sub(r'<span>\d+ / \d+</span>', f'<span>{pos:02d} / {total}</span>', html)
    (HTML / f"slide{pos:02d}.html").write_text(html, encoding="utf-8")

for old in HTML.glob("slide*.html"):
    n = int(old.stem[-2:])
    if n > total:
        old.unlink()

(ROOT / "build" / "notes.json").write_text(json.dumps(new_notes, indent=1), encoding="utf-8")

md = (ROOT / "presentation.md").read_text(encoding="utf-8")
md = md.replace("course: Computer Graphics and Multimedia, B.Tech 4th year", "course: Computer Graphics and Multimedia")
md = re.sub(r"## 10\.[^\n]*reproducing the base paper", "## 10. Reproducing the base paper", md, flags=re.I)
md = md.replace("Blended partition of unity and interface-aware fine-tuning for brick-decomposed neural volumes\n\nBase paper:", "Blended partition of unity and interface-aware fine-tuning for brick-decomposed neural volumes\n\n**Team:** Devyash Saini (202351030), Deepanshu (202351025), Himanshu (202351047)\n\nBase paper:")
old_table = '''| Comparison | Δ Global dB | Δ Boundary dB | Jump ratio | Seeds won |
|---|---|---|---|---|
| Ours vs g16 hard | +2.00 | +2.51 | 579× | 3/3 |
| Ours vs g4 hard | +0.51 | +1.44 | | |
| Blend only vs g4 hard | +0.20 | +1.24 | 463× | 3/3 |
| Ours vs g16 + recon FT | +1.50 | | | |
| Recon FT + blend vs g16 + recon FT | +1.73 | | | |'''
new_table = '''| Comparison | Global dB | Boundary dB | Render dB | Value jump | Grad jump | Wins |
|---|---:|---:|---:|---:|---:|---:|
| Ours vs g16 hard | +2.00 | +2.51 | +2.08 | 579× | 1.38× | 3/3 |
| g4 blend vs g16 hard | +1.69 | +2.32 | +1.16 | 567× | 1.36× | 3/3* |
| Ours vs g4 hard | +0.51 | +1.44 | +1.10 | 474× | 1.18× | 3/3 |
| Blend vs g4 hard | +0.20 | +1.24 | +0.19 | 463× | 1.16× | 3/3 |
| Ours vs g16 + recon FT | +1.50 | +1.77 | +1.08 | 445× | 1.32× | 1/1 |
| Recon FT + blend vs g16 + recon FT | +1.73 | +2.17 | +1.36 | 442× | 1.31× | 1/1 |
| Ours vs g16 32³ | −0.09 | +0.40 | −0.41 | 444× | 1.36× | 1/1 |
| Consistency FT vs recon FT | −0.21 | −0.38 | −0.25 | 1.01× | 1.01× | 0/2 |'''
md = md.replace(old_table, new_table).replace("The last two rows give the paper setting the same extra fine-tuning.", "Paired same-seed differences. *Render PSNR is won on 2/3 seeds.")

insertions = {
14: ("Full view: boundary slice error", "Implementation", "slice_error_g4_hard.png", "The full-resolution plot makes the boundary-aligned error visible."),
16: ("Full view: headline comparison", "Results", "headline.png", "Use the enlarged chart for the main quantitative comparison."),
20: ("Full view: capacity comparison", "Results", "capacity_curve.png", "The g16 capacity sweep is shown at full size."),
21: ("Full view: overlap-width comparison", "Results", "ghost_cell_curve.png", "The ghost-cell sweep is shown at full size."),
23: ("Full view: performance tradeoffs", "Results", "tradeoffs.png", "Throughput, frame time and memory are shown at full size."),
}
parts = re.split(r'(?=\n## \d+\. )', md)
front, slide_parts = parts[0], parts[1:]
expanded = []
for part in slide_parts:
    m = re.match(r'\n## (\d+)\.', part)
    orig = int(m.group(1))
    expanded.append(part)
    if orig in insertions:
        title, section, img, note = insertions[orig]
        expanded.append(f'''\n## 0. {title}\n*Section: {section}*\n\n![{title}](assets/{img})\n\n> **Speaker notes:** {note}\n\n---\n''')
md = front + ''.join(expanded)
counter = 0
def renum(match):
    global counter
    counter += 1
    return f"\n## {counter}. "
md = re.sub(r'\n## \d+\. ', renum, md)
(ROOT / "presentation.md").write_text(md, encoding="utf-8")

print(f"Wrote {total} HTML slides, {len(new_notes)} notes, and updated presentation.md")
