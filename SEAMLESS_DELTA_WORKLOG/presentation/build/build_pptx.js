const fs = require("fs"), path = require("path");
const { Presentation, PresentationFile } = require("@oai/artifact-tool");
(async () => {
  const notes = JSON.parse(fs.readFileSync(path.join(__dirname, "notes.json"), "utf8"));
  const p = Presentation.create({ slideSize: { width: 1280, height: 720 } });
  const slideCount = notes.length;
  for (let i = 1; i <= slideCount; i++) {
    const s = p.slides.add();
    const blob = fs.readFileSync(path.join(__dirname, "png", `slide${String(i).padStart(2, "0")}.png`));
    s.images.add({ blob, contentType: "image/png", fit: "cover", position: { left: 0, top: 0, width: 1280, height: 720 } });
    s.speakerNotes.text = notes[i - 1] || "";
  }
  const out = path.join(__dirname, "candidate_final.pptx");
  (await PresentationFile.exportPptx(p)).save(out);
  console.log("saved", out);
})().catch(e => { console.error(e); process.exit(1); });
