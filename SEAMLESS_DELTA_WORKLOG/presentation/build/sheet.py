from PIL import Image; import glob, os
os.chdir(r"C:\Users\Devyash\Documents\Computer_Graphics\SEAMLESS_DELTA_WORKLOG\presentation\build")
fs=sorted(glob.glob("png/slide*.png")); W,H=640,360
for k in range(0,len(fs),9):
    s=Image.new("RGB",(W*3,H*3),"white")
    for j,f in enumerate(fs[k:k+9]): s.paste(Image.open(f).convert("RGB").resize((W,H)),((j%3)*W,(j//3)*H))
    s.save(f"sheet{k//9+1}.png")