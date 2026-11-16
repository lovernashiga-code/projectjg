// Final-phase deck for "Face Recognition from Partial Face". Every number is read from the results files.
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");

const PROJ = process.argv[2];
const OUT = process.argv[3];
const R = path.join(PROJ, "results");
const FIGD = path.join(R, "figures");
const FIN = path.join(FIGD, "final");
const rd = (p) => JSON.parse(fs.readFileSync(p, "utf8").replace(/\bNaN\b/g, "null").replace(/-?Infinity/g, "null"));
const has = (p) => fs.existsSync(p);

const metrics = rd(path.join(R, "metrics.json"));
const targets = rd(path.join(R, "targets.json"));
const NUM = rd(path.join(R, "final_numbers.json"));
const stats = rd(path.join(R, "dataset_stats.json"));
const EX = NUM.example;
const ACC = {};
for (const r of metrics) ACC[`${r.method}|${r.variant}`] = r;
const acc = (m, v) => 100 * ACC[`${m}|${v}`].accuracy;
const f1 = (n, d = 1) => Number(n).toFixed(d);
const openSet = has(path.join(R, "open_set.json")) ? rd(path.join(R, "open_set.json")) : null;
const FEI = path.join(R, "fei");
const feiM = has(path.join(FEI, "metrics.json")) ? rd(path.join(FEI, "metrics.json")) : null;
const feiStats = has(path.join(FEI, "dataset_stats.json")) ? rd(path.join(FEI, "dataset_stats.json")) : null;
const VID = path.join(R, "video");
const video = has(path.join(VID, "video_check.json")) ? rd(path.join(VID, "video_check.json")) : null;
const videoM1 = has(path.join(VID, "M1", "video_check.json")) ? rd(path.join(VID, "M1", "video_check.json")) : null;
const CUSTD = path.join(R, "lfw+custom");
const CUST = has(path.join(CUSTD, "operating_point.json")) && has(path.join(CUSTD, "video_scores.json")) ? (() => {
  const st = rd(path.join(CUSTD, "dataset_stats.json"));
  const sessCount = {};
  for (const [n, ss] of Object.entries(st.sessions_per_class || {})) {
    sessCount[n] = {};
    for (const se of ss) {
      const dir = path.join(PROJ, "data", "custom", n, se);
      if (se && has(dir)) sessCount[n][se] = fs.readdirSync(dir).filter((f) => /\.(jpg|png)$/i.test(f)).length;
    }
  }
  const wdir = path.join(PROJ, "data", "unknown", "Worasak", "s1");
  return { stats: st, sessCount, op: rd(path.join(CUSTD, "operating_point.json")), acc: rd(path.join(CUSTD, "members_accuracy.json")),
    video: rd(path.join(CUSTD, "video_scores.json")), fig: path.join(CUSTD, "figures", "final"), demo: path.join(CUSTD, "figures", "demo"),
    raw: has(path.join(CUSTD, "video_raw.json")) ? rd(path.join(CUSTD, "video_raw.json")) : null,
    frames: has(path.join(CUSTD, "figures", "demo", "demo_frames.json")) ? rd(path.join(CUSTD, "figures", "demo", "demo_frames.json")) : {},
    worasakN: has(wdir) ? fs.readdirSync(wdir).length : 0, videoFile: path.join(VID, "members_M4", "group_1_annotated.mp4") };
})() : null;

const VARS = ["full", "upper_half", "lower_half", "eyes_only", "nose_mouth", "mask", "sunglasses"];
const VTH = { full: "เต็ม", upper_half: "ครึ่งบน", lower_half: "ครึ่งล่าง", eyes_only: "ตา", nose_mouth: "จมูก+ปาก",
  mask: "หน้ากาก", sunglasses: "แว่นกันแดด" };
const METH = ["M0", "M1", "M2", "M3", "M4"];
const MNAME = { M0: "Eigenfaces + SVM", M1: "VGG-Face · เทรนภาพเต็ม", M2: "VGG-Face · เต็ม+บางส่วน",
  M3: "FaceNet · เทรนภาพเต็ม", M4: "FaceNet · เต็ม+บางส่วน" };

// ===================== short final deck (5-minute talk + backup) =====================
const FONT = process.env.DECK_FONT || "TH SarabunPSK";
const K = FONT.includes("Sarabun") ? 1.38 : 1.0;          // Sarabun renders ~70% the size of Leelawadee
const z = (n) => Math.round(n * K * 2) / 2;
const HEX = { ink: "1F2430", muted: "5E636B", blue: "2A78D6", blueSoft: "E8F1FC", orange: "D95926",
  green: "2F9E44", red: "D03B3B", line: "DDE1E7", card: "FFFFFF", bg: "F4F6F9", grey: "A3A39E", dark: "141A26" };
const THEME = {
  name: "Partial Face", headFontFace: FONT, bodyFontFace: FONT,
  colors: { dk1: "1F2430", lt1: "FFFFFF", dk2: "141A26", lt2: "F4F6F9", accent1: "2A78D6", accent2: "D95926",
    accent3: "2F9E44", accent4: "D03B3B", accent5: "9CC2EE", accent6: "F4B08F", hlink: "2A78D6", folHlink: "6B6F76" },
};
const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.theme = { headFontFace: FONT, bodyFontFace: FONT };
pres.title = "Face Recognition from Partial Face — Final";
const W = 13.333, H = 7.5, MX = 0.6;
const FOOT = "FACE RECOGNITION FROM PARTIAL FACE · CP461 FINAL PROJECT";

pres.defineSlideMaster({
  title: "CONTENT", background: { color: HEX.bg },
  objects: [
    { text: { text: FOOT, options: { x: MX, y: 7.0, w: 7, h: 0.32, fontSize: z(9), color: HEX.muted, margin: 0, fontFace: FONT } } },
    { placeholder: { options: { name: "kicker", type: "body", x: MX, y: 0.32, w: 8.6, h: 0.36, fontSize: z(11), color: HEX.blue,
      bold: true, margin: 0, align: "left", fontFace: FONT }, text: "" } },
    { placeholder: { options: { name: "title", type: "title", x: MX, y: 0.66, w: W - 2 * MX, h: 0.78, fontSize: z(27),
      bold: true, color: HEX.ink, margin: 0, valign: "middle", align: "left", fontFace: FONT }, text: "" } },
  ],
  slideNumber: { x: W - MX - 0.8, y: 7.0, w: 0.8, h: 0.32, fontSize: z(9), color: HEX.muted, align: "right", fontFace: FONT },
});
pres.defineSlideMaster({
  title: "DARK", background: { color: HEX.dark },
  objects: [{ text: { text: FOOT, options: { x: MX, y: 7.0, w: 7, h: 0.32, fontSize: z(9), color: "8A93A6", margin: 0, fontFace: FONT } } }],
  slideNumber: { x: W - MX - 0.8, y: 7.0, w: 0.8, h: 0.32, fontSize: z(9), color: "8A93A6", align: "right", fontFace: FONT },
});

// ---------- helpers (font sizes are given at Leelawadee scale and multiplied by K) ----------
function pngSize(p) { const b = fs.readFileSync(p); return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) }; }
function img(s, p, x, y, w, h, align = "center") {
  if (!has(p)) { s.addText(`missing: ${path.basename(p)}`, { x, y, w, h, isTextBox: true, color: HEX.red, fontSize: 12 }); return; }
  const q = pngSize(p), r = Math.min(w / q.w, h / q.h), iw = q.w * r, ih = q.h * r;
  const ix = align === "left" ? x : x + (w - iw) / 2;
  s.addImage({ path: p, x: ix, y: y + (h - ih) / 2, w: iw, h: ih, altText: path.basename(p) });
}
function card(s, x, y, w, h, fill = HEX.card, line = HEX.line) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, fill: { color: fill }, line: { color: line, width: 0.75 }, rectRadius: 0.08 });
}
function scaleRuns(t) {
  return Array.isArray(t) ? t.map((r) => (r.options && r.options.fontSize ? { ...r, options: { ...r.options, fontSize: z(r.options.fontSize) } } : r)) : t;
}
function text(s, t, o) {
  const mono = o.fontFace === "Consolas";
  s.addText(mono ? t : scaleRuns(t), { isTextBox: true, margin: 0, color: HEX.ink, valign: "top", fontFace: FONT, ...o,
    fontSize: mono ? (o.fontSize || 12) : z(o.fontSize || 14) });
}
function table(s, rows, o) {
  const sc = (c) => (typeof c === "object" && c.options && c.options.fontSize ? { ...c, options: { ...c.options, fontSize: z(c.options.fontSize) } } : c);
  s.addTable(rows.map((r) => r.map(sc)), { fontFace: FONT, color: HEX.ink, border: { type: "solid", pt: 0.5, color: HEX.line },
    fill: { color: HEX.card }, margin: [1, 6, 1, 6], valign: "middle", ...o, fontSize: z(o.fontSize || 12) });
}
const HC = (t, al = "center") => ({ text: t, options: { bold: true, color: HEX.muted, fill: { color: HEX.bg }, align: al } });
let currentSection = "";
function section(t) { currentSection = t; pres.addSection({ title: t }); }
function content(kicker, title, accent, speaker) {
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: currentSection });
  s.addText(kicker, { placeholder: "kicker" });
  s.addText(accent ? [{ text: title }, { text: accent, options: { color: HEX.blue } }] : title, { placeholder: "title" });
  if (speaker) {
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: W - MX - 2.3, y: 0.3, w: 2.3, h: 0.38, fill: { color: HEX.blueSoft }, line: { color: HEX.blueSoft }, rectRadius: 0.08 });
    text(s, `ผู้นำเสนอ: ${speaker}`, { x: W - MX - 2.3, y: 0.3, w: 2.3, h: 0.38, fontSize: 10.5, bold: true, color: HEX.blue, align: "center", valign: "middle" });
  }
  return s;
}
function tag(s, t, x, y, color, w) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h: 0.34, fill: { color }, line: { color }, rectRadius: 0.06 });
  text(s, t, { x, y, w, h: 0.34, fontSize: 10.5, bold: true, color: "FFFFFF", align: "center", valign: "middle" });
}
const mean = (a) => a.reduce((x, y) => x + y, 0) / a.length;
const OCC = VARS.slice(1);
const M = (m, v, k) => ACC[`${m}|${v}`][k];
const occMean = (m, k) => mean(OCC.map((v) => M(m, v, k)));
const p = (x, d = 1) => f1(100 * x, d);

// ============ main slides ============
section("Presentation");

{ // 1 title
  const s = pres.addSlide({ masterName: "DARK", sectionTitle: currentSection });
  text(s, "CP461 COMPUTER VISION · FINAL PROJECT", { x: MX, y: 0.95, w: 8, h: 0.4, fontSize: 12, bold: true, color: "7FB0EE" });
  text(s, [{ text: "Face Recognition", options: { breakLine: true } }, { text: "from Partial Face", options: { color: "7FB0EE" } }],
    { x: MX, y: 1.45, w: 7.2, h: 2.3, fontSize: 48, bold: true, color: "FFFFFF" });
  text(s, "ระบบจดจำใบหน้าจากภาพที่ถูกบดบังบางส่วน (หน้ากาก แว่นกันแดด มือปิด) ด้วย MTCNN + VGG-Face / FaceNet และ Transfer Learning",
    { x: MX, y: 3.85, w: 6.7, h: 1.1, fontSize: 16, color: "C9D3E3" });
  text(s, "วรศักดิ์ แสนกล้า · ธีรภัทร ศิริมา · รพีพงศ์ สุบรรณวิลาศ · กฤต ไหลไผ่ทอง · อนุวัฒน์ เจริญสุข",
    { x: MX, y: 5.2, w: 6.8, h: 0.5, fontSize: 12.5, color: "8A93A6" });
  img(s, path.join(FIN, "example_all_variants.png"), 7.6, 1.3, 5.15, 1.25);
  img(s, path.join(CUST.demo, "M4_group_2_both.png"), 7.6, 2.8, 5.15, 2.9);
  text(s, "บน: ภาพทดสอบจริงจาก LFW 7 รูปแบบ · ล่าง: ระบบวิดีโอกับสมาชิกกลุ่ม", { x: 7.6, y: 5.8, w: 5.15, h: 0.4, fontSize: 10.5, color: "8A93A6" });
  s.addNotes("[วรศักดิ์] สวัสดีครับ กลุ่มเรานำเสนอโปรเจกต์ Face Recognition from Partial Face คือการจดจำว่าเป็นใคร แม้ใบหน้าถูกบังบางส่วน เช่น ใส่หน้ากาก ใส่แว่นกันแดด หรือเอามือปิด");
}

{ // 2 team
  const s = content("01 · TEAM", "สมาชิกและ", "บทบาทความรับผิดชอบ", "วรศักดิ์");
  const team = [["วรศักดิ์ แสนกล้า", "MODEL", "ออกแบบ Transfer Learning (VGG-Face, FaceNet) และชั้น Softmax"],
    ["ธีรภัทร ศิริมา", "DATA", "ชุดข้อมูล LFW, FEI และวิดีโอของกลุ่ม คัดจำนวนภาพต่อคลาส"],
    ["รพีพงศ์ สุบรรณวิลาศ", "PREPROCESSING", "MTCNN, Alignment และการสร้างภาพบางส่วน V0–V6"],
    ["กฤต ไหลไผ่ทอง", "PROTOTYPE", "Pipeline ทดสอบ: ภาพ เว็บแคม และระบบวิดีโอ"],
    ["อนุวัฒน์ เจริญสุข", "EVALUATION", "วัดผล (Accuracy, F1, ROC, Confusion) และเกณฑ์ τ"]];
  team.forEach((t, i) => {
    const x = MX + i * 2.45, y = 1.75;
    card(s, x, y, 2.27, 4.7);
    s.addShape(pres.shapes.OVAL, { x: x + 0.2, y: y + 0.25, w: 0.62, h: 0.62, fill: { color: HEX.blueSoft }, line: { color: HEX.blueSoft } });
    text(s, `0${i + 1}`, { x: x + 0.2, y: y + 0.25, w: 0.62, h: 0.62, align: "center", valign: "middle", bold: true, color: HEX.blue, fontSize: 14 });
    text(s, t[0], { x: x + 0.2, y: y + 1.05, w: 1.95, h: 0.9, fontSize: 15, bold: true });
    text(s, t[1], { x: x + 0.2, y: y + 2.0, w: 1.95, h: 0.4, fontSize: 10.5, bold: true, color: HEX.blue });
    text(s, t[2], { x: x + 0.2, y: y + 2.5, w: 1.95, h: 2.0, fontSize: 12.5, color: HEX.muted });
  });
  s.addNotes("[วรศักดิ์] สมาชิก 5 คน แบ่งงานตาม pipeline: ผมดูโมเดล ธีรภัทรดูข้อมูล รพีพงศ์ทำ preprocessing กฤตทำระบบทดสอบและวิดีโอ อนุวัฒน์วัดผลและเกณฑ์ความมั่นใจ");
}

{ // 3 problem + objectives
  const s = content("02 · PROBLEM & OBJECTIVES", "ใบหน้าถูกบดบัง", " = ระบบจดจำแม่นยำลดลง", "วรศักดิ์");
  text(s, "ระบบทั่วไปเทรนด้วยหน้าเต็ม แต่ในชีวิตจริงคนใส่หน้ากาก แว่นกันแดด หรือเอามือบัง ส่วนที่หายไปมักเป็นส่วนที่ใช้แยกคน",
    { x: MX, y: 1.62, w: 6.3, h: 0.95, fontSize: 15 });
  img(s, path.join(FIN, "example_all_variants.png"), MX, 2.6, 6.3, 1.25, "left");
  card(s, MX, 4.05, 6.3, 2.7, HEX.blueSoft, HEX.blueSoft);
  text(s, [{ text: "วัตถุประสงค์", options: { bold: true, color: HEX.blue, breakLine: true } },
    { text: "O1 สร้าง pipeline MTCNN → Align → VGG-Face/FaceNet → Softmax", options: { bullet: true, breakLine: true } },
    { text: "O2 วัดผลแยก 7 รูปแบบการบัง (V0–V6)", options: { bullet: true, breakLine: true } },
    { text: "O3 เทียบการเทรนด้วยหน้าเต็ม กับเทรนเพิ่มด้วยภาพบางส่วน และเทียบ Baseline", options: { bullet: true, breakLine: true } },
    { text: "O4 นำไปใช้กับวิดีโอจริงของกลุ่ม พร้อมตอบ Unknown กับคนแปลกหน้า", options: { bullet: true } }],
    { x: MX + 0.25, y: 4.15, w: 5.85, h: 2.5, fontSize: 13.5, paraSpaceAfter: 3 });
  card(s, 7.2, 1.65, 5.53, 5.1);
  text(s, "เป้าหมายวัดผลได้ (ตั้งก่อนทดลอง)", { x: 7.45, y: 1.82, w: 5.0, h: 0.4, fontSize: 13, bold: true, color: HEX.muted });
  [["T1", "Accuracy ใบหน้าเต็ม (M2)", "≥ 95%"], ["T2", "Accuracy หน้ากาก / แว่นกันแดด (M2)", "≥ 85%"],
    ["T3", "เทรนเพิ่มด้วยภาพบางส่วน ช่วยรูปแบบที่ยากที่สุด", "≥ +10 จุด"]].forEach((t, i) => {
    const y = 2.45 + i * 1.4;
    tag(s, t[0], 7.45, y + 0.12, HEX.blue, 0.6);
    text(s, t[1], { x: 8.2, y, w: 2.7, h: 1.0, fontSize: 13.5, valign: "middle" });
    text(s, t[2], { x: 10.9, y, w: 1.65, h: 1.0, fontSize: 22, bold: true, color: HEX.blue, align: "right", valign: "middle" });
  });
  s.addNotes("[วรศักดิ์] ปัญหาคือระบบทั่วไปเทรนด้วยหน้าเต็ม พอคนใส่หน้ากากหรือแว่น ความแม่นยำลดลงมาก เราจึงตั้งวัตถุประสงค์ 4 ข้อ และเป้าหมายที่วัดได้ 3 ข้อ: หน้าเต็มอย่างน้อย 95% หน้ากากและแว่นอย่างน้อย 85% และการเทรนด้วยภาพบางส่วนต้องช่วยกรณีที่ยากที่สุดอย่างน้อย 10 จุด");
}

{ // 4 data
  const s = content("03 · DATA", "ข้อมูลที่ใช้", " และตัวอย่างภาพแต่ละคลาส", "ธีรภัทร");
  const ds = [["LFW (การทดลองหลัก)", HEX.blue, `13,233 ภาพ 5,749 คน → คัดคนที่มี ≥ 20 ภาพ จำกัด ≤ 50 ภาพ/คน = ${stats.n_classes} คลาส ${stats.n_images.toLocaleString()} ภาพ · Train/Test 70/30 = ${NUM.train_images.toLocaleString()} / ${NUM.test_images}`],
    ["FEI (มุมหน้า)", HEX.orange, `200 คน × 14 ภาพ หมุนหน้า ~180° · MTCNN พบหน้า ${feiStats ? feiStats.n_images.toLocaleString() : "–"} ภาพ`],
    ["วิดีโอของกลุ่ม (ต้นแบบ)", HEX.green, `สมาชิก 3 คน ${Object.values(CUST.sessCount).reduce((a, o) => a + Object.values(o).reduce((x, y) => x + y, 0), 0)} เฟรมจากคลิปลงทะเบียน · คนแปลกหน้า 1 คน · คลิปทดสอบ 5 คลิป`]];
  ds.forEach((d, i) => {
    const y = 1.65 + i * 1.72;
    card(s, MX, y, 5.0, 1.58);
    tag(s, d[0], MX + 0.2, y + 0.18, d[1], 2.5);
    text(s, d[2], { x: MX + 0.2, y: y + 0.6, w: 4.6, h: 0.95, fontSize: 12 });
  });
  img(s, path.join(FIGD, "samples_per_class.png"), 5.85, 1.6, 6.9, 3.15);
  img(s, path.join(CUST.demo, "members_enrol.png"), 5.85, 4.85, 6.9, 1.95);
  text(s, "แหล่งข้อมูล: kaggle.com/datasets/jessicali9530/lfw-dataset · kaggle.com/datasets/whizzkid/fei-face-data · คลิปที่กลุ่มถ่ายเอง",
    { x: MX, y: 6.8, w: 5.1, h: 0.25, fontSize: 9, color: HEX.muted });
  s.addNotes("[ธีรภัทร] ข้อมูลหลักคือ LFW ภาพข่าวจริงที่แสงและมุมไม่ควบคุม เราคัดเฉพาะคนที่มีอย่างน้อย 20 ภาพ และจำกัดไม่เกิน 50 ภาพต่อคน ได้ 62 คลาส 2,063 ภาพ แบ่ง Train/Test 70/30 นอกจากนี้ใช้ FEI เพื่อดูผลของมุมหน้า และคลิปวิดีโอของสมาชิกกลุ่มสำหรับต้นแบบ ด้านขวาคือตัวอย่างภาพจริงแต่ละคลาส");
}

{ // 5 process
  const s = content("04 · PROCESS", "ขั้นตอนการทดลอง:", " 1 ภาพ → 7 รูปแบบ → ชื่อคน", "รพีพงศ์");
  const st = [["MTCNN", "กรอบ + 5 จุด", "example_mtcnn.png"], ["Alignment", "224×224 ตามแม่แบบ", "example_aligned.png"],
    ["Partial Face", "V0–V6", "example_variant.png"], ["Feature", "VGG-Face 4,096-d / FaceNet 512-d (แช่แข็ง)", "example_features.png"],
    ["Softmax", "ชื่อ + p · p < τ → Unknown", "example_top5_M2.png"]];
  st.forEach((t, i) => {
    const x = MX + i * 2.45;
    card(s, x, 1.62, 2.3, 2.75);
    text(s, `${i + 1}. ${t[0]}`, { x: x + 0.15, y: 1.7, w: 2.0, h: 0.4, fontSize: 14, bold: true });
    img(s, path.join(FIN, t[2]), x + 0.12, 2.12, 2.06, 1.5);
    text(s, t[1], { x: x + 0.15, y: 3.68, w: 2.0, h: 0.65, fontSize: 11, color: HEX.muted });
  });
  img(s, path.join(FIN, "variant_rules.png"), MX, 4.5, 9.3, 2.35, "left");
  card(s, 10.05, 4.5, 2.68, 2.35, HEX.blueSoft, HEX.blueSoft);
  text(s, [{ text: "ทำไม Align ก่อนบัง", options: { bold: true, breakLine: true } },
    { text: "ทุกภาพมีตา จมูก ปาก อยู่ที่เดียวกัน กล่องบัง V1–V6 จึงตกส่วนเดียวกันของทุกคน" }],
    { x: 10.2, y: 4.6, w: 2.4, h: 2.15, fontSize: 12 });
  s.addNotes("[รพีพงศ์] ทุกภาพผ่าน 5 ขั้น: MTCNN หาหน้าและจุด landmark 5 จุด จัดตำแหน่งเป็น 224×224 สร้างภาพบัง 7 รูปแบบ เช่น หน้ากาก แว่นกันแดด เหลือแต่ตา แล้วสกัด feature ด้วยโมเดลที่เทรนกับหน้ามาแล้ว และ Softmax ทายชื่อ ถ้าความมั่นใจต่ำกว่า τ ตอบ Unknown ภาพทุกช่องเป็นภาพทดสอบจริง");
}

{ // 6 methods + metrics
  const s = content("05 · METHODS & METRICS", "เทียบ 5 วิธี", " และตัววัดประสิทธิภาพ", "วรศักดิ์");
  const rows = [["วิธี", "Feature (แช่แข็ง)", "เทรนด้วย"].map((h) => HC(h, "left"))];
  [["M0", "Eigenfaces (PCA 150) + SVM", "V0 · baseline ดั้งเดิม"], ["M1", "VGG-Face fc7 · 4,096-d", "V0 หน้าเต็ม"],
    ["M2", "VGG-Face fc7 · 4,096-d", "V0–V6 (7 เท่า)"], ["M3", "FaceNet (VGGFace2) · 512-d", "V0 หน้าเต็ม"],
    ["M4", "FaceNet (VGGFace2) · 512-d", "V0–V6 (7 เท่า)"]].forEach((r) => rows.push([{ text: r[0], options: { bold: true, color: HEX.blue } }, r[1], r[2]]));
  table(s, rows, { x: MX, y: 1.7, w: 7.0, colW: [0.8, 3.4, 2.8], rowH: 0.5, fontSize: 12.5 });
  card(s, MX, 4.9, 7.0, 1.85, HEX.blueSoft, HEX.blueSoft);
  text(s, "Transfer Learning: น้ำหนัก VGG-Face / FaceNet แช่แข็ง เทรนเฉพาะชั้น Softmax (LogisticRegression) · ทดสอบชุด Test เดียวกันทุกวิธี แยกทีละรูปแบบ V0–V6 · seed 42",
    { x: MX + 0.25, y: 5.0, w: 6.5, h: 1.65, fontSize: 13, valign: "middle" });
  const mt = [["Accuracy", "ทายชื่อถูกกี่ % (ตัววัดหลัก, T1–T3)"], ["Precision / Recall / F1 (macro)", "ทายผิดคน / จำไม่ได้ เฉลี่ยทุกคลาส"],
    ["ROC-AUC (OvR)", "แยกคนถูก/ผิดได้ดีแค่ไหนทุกเกณฑ์"], ["Confusion Matrix", "สับสนระหว่างคนไหน"], ["Coverage@τ / Unknown", "ตอบชื่อเมื่อมั่นใจพอ กันคนแปลกหน้า"]];
  card(s, 7.85, 1.7, 4.88, 5.05);
  mt.forEach((m, i) => {
    const y = 1.9 + i * 0.96;
    text(s, m[0], { x: 8.05, y, w: 4.5, h: 0.4, fontSize: 13.5, bold: true });
    text(s, m[1], { x: 8.05, y: y + 0.4, w: 4.5, h: 0.45, fontSize: 11.5, color: HEX.muted });
  });
  s.addNotes("[วรศักดิ์] เราเทียบ 5 วิธี: M0 คือ Eigenfaces + SVM เป็น baseline แบบดั้งเดิม M1 และ M3 ใช้ VGG-Face และ FaceNet เทรนด้วยหน้าเต็มอย่างเดียว ส่วน M2 และ M4 เทรนเพิ่มด้วยภาพบัง 6 แบบ วัดผลด้วย Accuracy, Precision, Recall, F1, ROC-AUC และ confusion matrix");
}

{ // 7 results table + chart
  const s = content("06 · RESULTS", "ผลการทดลองจริงบน LFW:", " 5 วิธี", "อนุวัฒน์");
  const best = (k, f) => Math.max(...METH.map((m) => f(m, k)));
  const cols = [["Acc เต็ม", (m) => M(m, "full", "accuracy")], ["Acc", (m) => occMean(m, "accuracy")], ["Prec.", (m) => occMean(m, "precision")],
    ["Recall", (m) => occMean(m, "recall")], ["F1", (m) => occMean(m, "f1")], ["AUC", (m) => occMean(m, "auc")]];
  const SH = { M0: "Eigenfaces + SVM", M1: "VGG-Face เต็ม", M2: "VGG-Face เต็ม+บางส่วน", M3: "FaceNet เต็ม", M4: "FaceNet เต็ม+บางส่วน" };
  const rows = [[HC("วิธี", "left"), HC("V0"), { text: "เฉลี่ย 6 รูปแบบที่ถูกบัง V1–V6 (macro)", options: { colspan: 5, bold: true, color: HEX.muted, fill: { color: HEX.bg }, align: "center" } }],
    [HC(""), HC(""), ...cols.slice(1).map((c) => HC(c[0]))]];
  METH.forEach((m) => rows.push([{ text: `${m} ${SH[m]}`, options: { bold: true } },
    ...cols.map(([, f], j) => {
      const v = f(m), b = Math.max(...METH.map((q) => cols[j][1](q)));
      return { text: j === 5 ? f1(v, 3) : p(v), options: { align: "center", bold: v === b, color: v === b ? HEX.blue : HEX.ink } };
    })]));
  table(s, rows, { x: MX, y: 1.6, w: 7.3, colW: [2.42, 0.82, 0.82, 0.82, 0.82, 0.8, 0.8], rowH: 0.37, fontSize: 11 });
  text(s, `T1 ${p(targets.T1.value)}% ✓ · T2 หน้ากาก ${p(targets.T2.value.mask)}% / แว่น ${p(targets.T2.value.sunglasses)}% ✓ · T3 ${targets.T3.M1_worst_variant.map((v) => `${VTH[v]} +${p(targets.T3.value[v])}`).join(", ")} จุด ✓`,
    { x: MX, y: 4.62, w: 7.3, h: 0.4, fontSize: 12.5, bold: true, color: HEX.green });
  card(s, MX, 5.1, 7.3, 1.65, HEX.blueSoft, HEX.blueSoft);
  text(s, [{ text: "ผ่านเป้าหมาย T1–T3 ทุกข้อ", options: { bold: true, breakLine: true } },
    { text: `M2/M4 (เทรนเพิ่มด้วยภาพบางส่วน) ดีกว่า M1/M3 ทุกตัววัดเมื่อถูกบัง · M4 ดีที่สุดในค่าเฉลี่ย · M0 แทบใช้ไม่ได้เมื่อถูกบัง (${p(occMean("M0", "accuracy"))}%)` }],
    { x: MX + 0.25, y: 5.15, w: 6.8, h: 1.55, fontSize: 12.5, valign: "middle" });
  const data = METH.map((m) => ({ name: m, labels: VARS.map((v, i) => `V${i} ${VTH[v]}`), values: VARS.map((v) => +f1(acc(m, v))) }));
  s.addChart(pres.charts.BAR, data, { x: 8.1, y: 1.55, w: 4.63, h: 5.25, barDir: "bar", barGrouping: "clustered", barGapWidthPct: 35,
    chartColors: ["A3A39E", "9CC2EE", "2A78D6", "F4B08F", "D95926"], valAxisMaxVal: 100, valAxisMinVal: 0, catAxisOrientation: "maxMin",
    valAxisLabelFontSize: z(9), catAxisLabelFontSize: z(10), valAxisLabelColor: HEX.muted, catAxisLabelColor: HEX.ink,
    catAxisLabelFontFace: FONT, valAxisLabelFontFace: FONT, legendFontFace: FONT, titleFontFace: FONT,
    valGridLine: { color: "E3E6EA", size: 0.5 }, catGridLine: { style: "none" }, showLegend: true, legendPos: "b", legendFontSize: z(10),
    showTitle: true, title: "Accuracy (%) แยกตามรูปแบบที่ทดสอบ", titleFontSize: z(11), titleColor: HEX.ink });
  s.addNotes("[อนุวัฒน์] ตารางนี้คือผลจริงของ 5 วิธี ช่องแรกคือหน้าเต็ม ช่องที่เหลือเป็นค่าเฉลี่ยของ 6 รูปแบบที่ถูกบัง ทั้ง Accuracy, Precision, Recall, F1 และ AUC ทุกวิธีที่เทรนเพิ่มด้วยภาพบางส่วนดีกว่าแบบเทรนหน้าเต็มอย่างชัดเจน และผ่านเป้าหมาย T1 ถึง T3 ครบ กราฟด้านขวาคือ Accuracy แยกแต่ละรูปแบบ");
}

{ // 8 confusion + ROC
  const s = content("06 · RESULTS — GRAPHS", "Confusion Matrix และ ROC:", " M1 เทียบ M2 เมื่อใส่หน้ากาก", "อนุวัฒน์");
  img(s, path.join(FIN, "cm_top12_mask_M1_vs_M2.png"), MX, 1.55, 8.4, 4.7, "left");
  img(s, path.join(FIGD, "roc_mask_methods.png"), 9.15, 1.55, 3.58, 3.4);
  card(s, 9.15, 5.0, 3.58, 1.75, HEX.blueSoft, HEX.blueSoft);
  text(s, `M1 สับสนกระจายหลายคู่ (ตัวเลขแดงนอกเส้นทแยง) · M2 แทบเหลือแต่เส้นทแยง · Accuracy ${p(M("M1", "mask", "accuracy"))}% → ${p(M("M2", "mask", "accuracy"))}% · AUC ${f1(M("M1", "mask", "auc"), 3)} → ${f1(M("M2", "mask", "auc"), 3)}`,
    { x: 9.3, y: 5.08, w: 3.3, h: 1.6, fontSize: 11.5, valign: "middle" });
  text(s, "ตัวเลข = จำนวนภาพ Test · แสดง 12 คนที่ M1 สับสนบ่อยที่สุด + คอลัมน์ \"อื่น ๆ\" · confusion matrix ครบทุกการทดลองอยู่ภาคผนวก",
    { x: MX, y: 6.4, w: 8.4, h: 0.35, fontSize: 10, color: HEX.muted });
  s.addNotes("[อนุวัฒน์] ตัวอย่าง confusion matrix เมื่อใส่หน้ากาก ด้านซ้าย M1 ที่เทรนหน้าเต็มอย่างเดียว ทายสลับคนหลายคู่ ด้านขวา M2 ที่เทรนเพิ่มด้วยภาพบัง เหลือแทบแต่เส้นทแยงซึ่งคือทายถูก กราฟ ROC ด้านขวาก็ยืนยันว่า M2 แยกคนได้ดีกว่า");
}

{ // 9 analysis + right/wrong
  const s = content("06 · ANALYSIS", "วิเคราะห์ผล", " และตัวอย่างที่ทายถูก / ผิด", "รพีพงศ์");
  const pts = [["เทรนด้วยภาพบางส่วนช่วยมากที่สุด", `ครึ่งล่าง M1 ${p(M("M1", "lower_half", "accuracy"))}% → M2 ${p(M("M2", "lower_half", "accuracy"))}% · จมูก+ปาก ${p(M("M1", "nose_mouth", "accuracy"))}% → ${p(M("M2", "nose_mouth", "accuracy"))}%`],
    ["ดวงตาสำคัญที่สุด", `ปิดตา (แว่น) แย่กว่าปิดปาก (หน้ากาก): M1 แว่น ${p(M("M1", "sunglasses", "accuracy"))}% vs หน้ากาก ${p(M("M1", "mask", "accuracy"))}%`],
    ["FaceNet ทนกว่า VGG-Face", `เมื่อเทรนหน้าเต็มอย่างเดียว M3 ดีกว่า M1 ทุกรูปแบบที่ถูกบัง · M4 ดีที่สุดกับแว่น ${p(M("M4", "sunglasses", "accuracy"))}%`]];
  pts.forEach((t, i) => {
    const x = MX + i * 4.1;
    card(s, x, 1.62, 3.95, 1.95);
    text(s, t[0], { x: x + 0.2, y: 1.72, w: 3.55, h: 0.45, fontSize: 14, bold: true, color: HEX.blue });
    text(s, t[1], { x: x + 0.2, y: 2.2, w: 3.55, h: 1.3, fontSize: 12 });
  });
  img(s, path.join(FIN, "real_predictions_lower_half.png"), MX, 3.7, W - 2 * MX, 2.75);
  text(s, "ภาพ Test จริงเมื่อเห็นเฉพาะครึ่งล่าง (V2) · กรอบเขียว/แดง = M2 ถูก/ผิด · ใต้ภาพ = คำตอบจริงของ M1 / M2 พร้อม p · คัดมา 4 ภาพที่ M1 ผิด M2 ถูก, 2 ภาพถูกทั้งคู่, 2 ภาพที่ M2 ผิด",
    { x: MX, y: 6.5, w: W - 2 * MX, h: 0.35, fontSize: 10, color: HEX.muted });
  s.addNotes("[รพีพงศ์] สิ่งที่เราพบคือ การให้โมเดลเห็นภาพที่ถูกบังตอนเทรนช่วยมากที่สุด เช่นครึ่งล่างดีขึ้นเกือบ 50 จุด ดวงตาเป็นส่วนที่สำคัญที่สุด การปิดตาจึงยากกว่าการปิดปาก และ FaceNet ทนต่อการบังมากกว่า VGG-Face ด้านล่างคือตัวอย่างจริงที่ทายถูกและผิด กรณีที่ผิดมักเป็นภาพหันข้างหรือแสงไม่ดี");
}

{ // 10 prototype
  const s = content("07 · PROTOTYPE", "ต้นแบบ: ระบบวิดีโอ", " ระบุชื่อสมาชิกที่ถูกบัง", "กฤต");
  if (has(CUST.videoFile)) {
    const cover = "image/png;base64," + fs.readFileSync(path.join(CUST.demo, "M4_group_1_stranger.png")).toString("base64");
    s.addMedia({ type: "video", path: CUST.videoFile, x: MX, y: 1.55, w: 7.3, h: 4.11, cover });
  }
  text(s, "คลิปทดสอบ (ไม่ใช้เทรน): รพีพงศ์เอามือปิดปาก · กฤตใส่หน้ากาก+แว่น · วรศักดิ์ (คนแปลกหน้า) เดินเข้ามา · กรอบ + ลูกศร + ชื่อ + % · คลิกเพื่อเล่น",
    { x: MX, y: 5.72, w: 7.3, h: 0.6, fontSize: 11, color: HEX.muted });
  text(s, "MTCNN → Align → Feature → Softmax → ติดตามหน้าข้ามเฟรม → โหวต 15 เฟรม → วาดผล", { x: MX, y: 6.3, w: 7.3, h: 0.45, fontSize: 12, bold: true, color: HEX.blue });
  const vs = CUST.video, per = {};
  for (const m of ["M2", "M4"]) for (const r of vs[m]) {
    per[r.truth] = per[r.truth] || { M2: [0, 0], M4: [0, 0] }; per[r.truth][m][0] += r.ok ? 1 : 0; per[r.truth][m][1] += 1;
  }
  const rows = [[HC("คลิปทดสอบ 5 คลิป (track)", "left"), HC("M2"), HC("M4")]];
  ["Rapepong", "Threeraphat", "Krit", "Worasak"].forEach((k) => {
    const c = (m) => ({ text: `${per[k][m][0]}/${per[k][m][1]}`, options: { align: "center", bold: true, color: per[k][m][0] === per[k][m][1] ? HEX.green : HEX.red } });
    rows.push([k === "Worasak" ? { text: "Worasak (คนแปลกหน้า) → Unknown", options: { bold: true, color: HEX.red } } : { text: k, options: { bold: true } }, c("M2"), c("M4")]);
  });
  table(s, rows, { x: 8.15, y: 1.6, w: 4.58, colW: [2.98, 0.8, 0.8], rowH: 0.42, fontSize: 12 });
  const ma = CUST.acc;
  card(s, 8.15, 3.95, 4.58, 2.8, HEX.blueSoft, HEX.blueSoft);
  text(s, [{ text: "สมาชิกใน session ที่โมเดลไม่เคยเห็น (M4)", options: { bold: true, breakLine: true } },
    { text: `หน้าเต็ม ${p(ma.M4.full)}% · หน้ากาก ${p(ma.M4.mask)}% · แว่น ${p(ma.M4.sunglasses)}%`, options: { color: HEX.blue, bold: true, breakLine: true } },
    { text: " ", options: { breakLine: true, fontSize: 5 } },
    { text: `M4 ไม่เคยเรียกผิดคน (ที่พลาดตอบ Unknown) · M2 ตอบชื่อได้มากกว่า แต่เรียกคนแปลกหน้าเป็นสมาชิก 1 ครั้ง · สมาชิก 3 คน + คนแปลกหน้า 1 คน ถ่ายช่วงเวลาใกล้กัน = ระดับเดโม` }],
    { x: 8.35, y: 4.05, w: 4.2, h: 2.6, fontSize: 12, paraSpaceAfter: 2 });
  s.addNotes("[กฤต] เรานำโมเดลไปใช้กับวิดีโอจริงของกลุ่ม สมาชิกลงทะเบียนด้วยคลิปหน้าเต็ม 2 คลิป แล้วทดสอบกับคลิปที่ใส่หน้ากาก ใส่แว่น และเอามือปิด ระบบวาดกรอบ ลูกศร และชื่อของทุกคน คนแปลกหน้าขึ้น Unknown ตารางด้านขวาสรุปผลทุกคลิป โมเดล FaceNet ไม่เคยเรียกผิดคน ส่วน VGG-Face ตอบชื่อได้มากกว่าแต่เคยเรียกคนแปลกหน้าเป็นสมาชิกหนึ่งครั้ง");
}

{ // 11 conclusion
  const s = content("08 · CONCLUSION", "สรุปและ", "ข้อเสนอแนะ", "ธีรภัทร");
  const boxes = [
    ["โมเดลตอบโจทย์ที่ตั้งไว้หรือไม่", [`ตอบโจทย์: ผ่าน T1–T3 ครบ (M2 หน้าเต็ม ${p(targets.T1.value)}%, หน้ากาก ${p(targets.T2.value.mask)}%, แว่น ${p(targets.T2.value.sunglasses)}%)`,
      "ใช้กับวิดีโอจริงได้ ระบุสมาชิกที่ถูกบังและตอบ Unknown กับคนแปลกหน้า"]],
    ["ได้โมเดลที่ดีกว่า Paper หรือไม่", null],
    ["จุดแข็ง / จุดอ่อน", ["แข็ง: ทนการบังมาก, เทรนเร็ว (เทรนแค่ Softmax), ต่อเติมกับวิดีโอได้",
      "อ่อน: MTCNN หาหน้าไม่เจอเมื่อบังทั้งตาและปากหรือหันข้าง · คนแปลกหน้าที่หน้าคล้ายสมาชิกหลุดได้ · การบังในการทดลองเป็นกล่องบนภาพ"]],
    ["แนวทางพัฒนาต่อ", ["เพิ่มคนแปลกหน้าหลายคน และใช้ระยะ cosine แทน softmax", "detector ที่ทนการบัง · ทดสอบหน้ากากจริงจำนวนมากขึ้น · ลงทะเบียนหลายวัน/หลายสภาพ"]]];
  boxes.forEach((b, i) => {
    const x = MX + (i % 2) * 6.12, y = 1.6 + Math.floor(i / 2) * 2.62;
    card(s, x, y, 5.97, 2.48);
    text(s, b[0], { x: x + 0.22, y: y + 0.12, w: 5.5, h: 0.45, fontSize: 15, bold: true, color: i === 2 ? HEX.orange : HEX.blue });
    if (b[1]) {
      text(s, b[1].map((t, j) => ({ text: t, options: { bullet: true, breakLine: j < b[1].length - 1 } })), { x: x + 0.22, y: y + 0.62, w: 5.55, h: 1.8, fontSize: 12.5, paraSpaceAfter: 3 });
      return;
    }
    // paper [1] Table I (LFW, ~2,800 images, 70/30, MTCNN + VGG-Face + Softmax) vs our M2 / M4 on the same face parts
    const PAPER = { full: 98, upper_half: 90, lower_half: 83 };
    const rows = [[HC("LFW · Accuracy (%)", "left"), HC("Paper [1]"), HC("M2 (เรา)"), HC("M4 (เรา)")]];
    [["full", "หน้าเต็ม"], ["upper_half", "ครึ่งบน"], ["lower_half", "ครึ่งล่าง"]].forEach(([v, l]) => {
      const ours = ["M2", "M4"].map((m) => 100 * M(m, v, "accuracy"));
      rows.push([l, { text: String(PAPER[v]), options: { align: "center" } },
        ...ours.map((o) => ({ text: f1(o), options: { align: "center", bold: true, color: o > PAPER[v] ? HEX.green : HEX.red } }))]);
    });
    table(s, rows, { x: x + 0.22, y: y + 0.6, w: 5.53, colW: [1.85, 1.2, 1.24, 1.24], rowH: 0.3, fontSize: 11 });
    text(s, "ดีกว่าทุกส่วนที่เทียบได้ แต่ชุดภาพ/จำนวนคน/เส้นแบ่งครึ่งหน้าไม่เหมือนกัน · Paper ไม่ได้ทดสอบหน้ากาก/แว่น (เราทดสอบเพิ่ม)",
      { x: x + 0.22, y: y + 1.88, w: 5.55, h: 0.55, fontSize: 10.5, color: HEX.muted });
  });
  s.addNotes("[ธีรภัทร] สรุปคือโมเดลตอบโจทย์ ผ่านเป้าหมายทั้ง 3 ข้อ และใช้กับวิดีโอจริงได้ เทียบกับ paper ต้นแบบที่ใช้ LFW และ VGG-Face เหมือนกัน paper ได้หน้าเต็ม 98 ครึ่งบน 90 ครึ่งล่าง 83 เปอร์เซ็นต์ ของเราดีกว่าทุกส่วน เพราะเราเทรนเพิ่มด้วยภาพที่ถูกบัง แต่ชุดภาพและจำนวนคนไม่เหมือนกันทีเดียว และเราทดสอบหน้ากากกับแว่นเพิ่มซึ่ง paper ไม่ได้ทำ จุดอ่อนคือถ้าบังทั้งตาและปาก ระบบหาหน้าไม่เจอ และคนแปลกหน้าที่หน้าคล้ายสมาชิกอาจหลุด แนวทางต่อไปคือเพิ่มคนแปลกหน้า และใช้ระยะ cosine แทน softmax");
}

{ // 12 Q&A + references
  const s = pres.addSlide({ masterName: "DARK", sectionTitle: currentSection });
  text(s, "Q & A", { x: MX, y: 1.4, w: 5.5, h: 1.4, fontSize: 64, bold: true, color: "FFFFFF" });
  text(s, "ขอบคุณครับ", { x: MX, y: 2.85, w: 5.5, h: 0.9, fontSize: 34, bold: true, color: "7FB0EE" });
  text(s, "Code: README.md · src/ · ผลทั้งหมดใน results/", { x: MX, y: 3.9, w: 5.5, h: 0.5, fontSize: 13, color: "8A93A6" });
  const refs = ["[1] Alfattama, Kanungo & Bisoy (2021). Face Recognition from Partial Face Data. APSIT 2021, IEEE. doi:10.1109/APSIT52773.2021.9641286",
    "[2] Zhang et al. (2016). MTCNN. IEEE SPL 23(10)", "[3] Parkhi, Vedaldi & Zisserman (2015). Deep Face Recognition. BMVC",
    "[4] Cao et al. (2018). VGGFace2. IEEE FG", "[5] Huang et al. (2007). Labeled Faces in the Wild. UMass TR 07-49",
    "[6] Thomaz & Giraldi (2010). FEI Face Database. Image and Vision Computing 28(6)",
    "Code: github.com/rcmalli/keras-vggface · github.com/timesler/facenet-pytorch · scikit-learn · OpenCV"];
  text(s, refs.map((r, i) => ({ text: r, options: { breakLine: i < refs.length - 1 } })), { x: 6.6, y: 1.3, w: 6.13, h: 5.4, fontSize: 10.5, color: "C9D3E3", paraSpaceAfter: 5 });
  s.addNotes("[ทุกคน] ขอบคุณครับ ยินดีตอบคำถาม (สไลด์ถัดไปเป็นภาคผนวก ใช้เปิดตอนตอบคำถาม)");
}

// ============ backup ============
section("ภาคผนวก (ตอบคำถาม)");

{ // A1 feedback
  const s = content("ภาคผนวก A · FEEDBACK → FINAL", "สิ่งที่ปรับจาก", " Progress Presentation");
  const rows = [["ข้อเสนอแนะ", "สิ่งที่ทำ", "ดูที่"].map((h) => HC(h, "left"))];
  [["confusion matrix ของการทดลองจริง ทุกการทดลอง", "LFW 5 วิธี × 7 รูปแบบ = 35 ชุด + ขยาย M1 vs M2 + ของสมาชิก", "หน้า 8, A5, A6"],
    ["ผลจริงของการปรับรูป: คน 1 คน ปิดหน้า / เห็นบางส่วน", "ภาพจริงคนเดียวกัน V0–V6 พร้อมพิกัดกล่องจากโค้ด", "หน้า 3, 5"],
    ["V0–V6 ชุดข้อมูลเป็นอย่างไร", "ทุกชุด Train 1,444 / Test 619 ภาพเดิม ต่างแค่ส่วนที่บัง", "หน้า 4, 5"],
    ["มีแต่ภาพแนวคิด ควรแสดงภาพจริง", "ทุกภาพในสไลด์เป็นภาพทดสอบจริง + ตัวอย่างถูก/ผิด + วิดีโอ", "หน้า 5, 9, 10"],
    ["ไม่ได้อธิบายโค้ด", "โครงสร้างโค้ดและโค้ดหลัก", "A2"]].forEach((r) => rows.push([r[0], { text: r[1], options: { bold: true } }, { text: r[2], options: { color: HEX.blue, bold: true } }]));
  table(s, rows, { x: MX, y: 1.7, w: W - 2 * MX, colW: [4.4, 6.0, 1.73], rowH: [0.45, 0.8, 0.8, 0.8, 0.8, 0.8], fontSize: 13 });
}

{ // A2 code
  const s = content("ภาคผนวก A2 · CODE", "โครงสร้างโค้ด", " (1 ไฟล์ = 1 ขั้นตอน)");
  const files = [["common.py", "align(), VARIANTS + apply_variant()"], ["step1_detect_align.py", "เลือกคน → MTCNN → align"],
    ["step2_extract_features.py", "V0–V6 → VGG-Face / FaceNet features"], ["step3_train_evaluate.py", "split → เทรน M0–M4 → วัดผล → τ"],
    ["step4/5_figures.py", "กราฟ ROC, confusion ทุกการทดลอง"], ["recognizer.py · video.py", "ใช้งานจริง: ภาพ / วิดีโอ + tracking"],
    ["members_eval.py", "ผลของสมาชิก + ให้คะแนนวิดีโอ"]];
  files.forEach((f, i) => {
    const y = 1.65 + i * 0.73;
    card(s, MX, y, 5.6, 0.63);
    text(s, f[0], { x: MX + 0.12, y, w: 2.85, h: 0.63, fontSize: 10, bold: true, fontFace: "Consolas", color: HEX.blue, valign: "middle" });
    text(s, f[1], { x: MX + 3.0, y, w: 2.55, h: 0.63, fontSize: 11, valign: "middle" });
  });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 6.45, y: 1.65, w: 6.28, h: 5.1, fill: { color: "1E2433" }, line: { color: "1E2433" }, rectRadius: 0.06 });
  const code = `# common.py: V1-V4 keep a box, V5-V6 hide one
def apply_variant(img, name):
    mode, box = VARIANTS[name]
    if box is None:
        return img.copy()
    x0, y0, x1, y1 = box
    if mode == "keep":
        out = np.zeros_like(img)
        out[y0:y1, x0:x1] = img[y0:y1, x0:x1]
    else:
        out = img.copy()
        out[y0:y1, x0:x1] = 0
    return out

# step3: M1 = full only, M2 = V0-V6
X = [feats[v][idx_train] for v in used]
clf = LogisticRegression(C=10).fit(
    np.concatenate(X),
    np.tile(y[idx_train], len(X)))

# recognizer: Unknown below tau
name = best if p >= tau else "Unknown"`;
  s.addText(code.split("\n").map((ln, i, a) => ({ text: ln || " ", options: { color: ln.trim().startsWith("#") ? "8FA1B3" : "E6EAF0", breakLine: i < a.length - 1 } })),
    { x: 6.65, y: 1.78, w: 5.95, h: 4.9, isTextBox: true, margin: 0, fontFace: "Consolas", fontSize: 10.5, valign: "top" });
}

{ // A3 stranger / tau
  const s = content("ภาคผนวก A3 · UNKNOWN & τ", "คนแปลกหน้า", " กับเกณฑ์ความมั่นใจ τ");
  img(s, path.join(FIGD, "open_set_M2.png"), MX, 1.55, 6.4, 3.6, "left");
  const op = CUST.op;
  const rows = [[HC("สมาชิก (session ที่ไม่เคยเห็น) / คนแปลกหน้า", "left"), HC(`M2 τ ${f1(op.M2.tau, 2)}`), HC(`M4 τ ${f1(op.M4.tau, 2)}`)]];
  [["full", "เต็ม"], ["mask", "หน้ากาก"]].forEach(([v, l]) => {
    rows.push([`${l}: สมาชิกถูก + ตอบชื่อ`, p(op.M2[v].members_right_at_tau), p(op.M4[v].members_right_at_tau)]);
    rows.push([`${l}: Worasak ถูกเรียกเป็นสมาชิก`, p(op.M2[v].stranger_as_member_at_tau), p(op.M4[v].stranger_as_member_at_tau)]);
  });
  table(s, rows, { x: 7.25, y: 1.6, w: 5.48, colW: [3.48, 1.0, 1.0], rowH: 0.48, fontSize: 11.5 });
  card(s, MX, 5.3, W - 2 * MX, 1.45, HEX.blueSoft, HEX.blueSoft);
  text(s, `τ เลือกจากคนแปลกหน้า LFW 500 คน (ผ่าน ≤ 5%) · Worasak ไม่ได้ใช้เลือก τ ใช้ทดสอบอย่างเดียว · คนดัง LFW แยกง่าย แต่เพื่อนร่วมชั้นที่ใส่แว่นเหมือนสมาชิกยากกว่ามาก → ต้องมีคนแปลกหน้าที่หน้าคล้ายกลุ่มมากขึ้น`,
    { x: MX + 0.25, y: 5.38, w: W - 2 * MX - 0.5, h: 1.3, fontSize: 12.5, valign: "middle" });
}

if (feiM) { // A4 FEI
  const s = content("ภาคผนวก A4 · FEI", "FEI:", " มุมหน้ากระทบผลอย่างไร");
  const fa = {}; for (const r of feiM) fa[`${r.method}|${r.variant}`] = 100 * r.accuracy;
  img(s, path.join(FEI, "figures", "pose_fei.png"), MX, 1.55, 7.2, 5.2, "left");
  const rows = [["FEI Accuracy (%)", "M1", "M2", "M4"].map((h, i) => HC(h, i ? "center" : "left"))];
  VARS.forEach((v, i) => rows.push([`V${i} ${VTH[v]}`, ...["M1", "M2", "M4"].map((m) => ({ text: f1(fa[`${m}|${v}`]), options: { align: "center" } }))]));
  table(s, rows, { x: 8.1, y: 1.6, w: 4.63, colW: [1.9, 0.91, 0.91, 0.91], rowH: 0.42, fontSize: 12 });
  text(s, `${feiStats.n_classes} คน · MTCNN พบหน้า ${feiStats.n_images.toLocaleString()} ภาพ (ข้าม ${feiStats.skipped_no_face} ภาพหน้าด้านข้าง) · ผ่าน T1–T3 เช่นกัน`,
    { x: 8.1, y: 5.1, w: 4.63, h: 1.2, fontSize: 12 });
}

{ // A5 confusion grid
  const s = content("ภาคผนวก A5 · CONFUSION MATRIX", "Confusion matrix ครบ 35 การทดลอง", ` (LFW ${stats.n_classes}×${stats.n_classes})`);
  img(s, path.join(FIN, "confusion_grid.png"), MX, 1.5, W - 2 * MX, 5.35);
}

{ // A6 members confusion
  const s = content("ภาคผนวก A6 · สมาชิกกลุ่ม", "Confusion matrix ของสมาชิก:", " session ที่โมเดลไม่เคยเห็น");
  img(s, path.join(CUST.fig, "members_confusion_M4.png"), MX, 1.5, W - 2 * MX, 2.6);
  img(s, path.join(CUST.fig, "members_confusion_M2.png"), MX, 4.15, W - 2 * MX, 2.6);
  text(s, "บน M4 FaceNet · ล่าง M2 VGG-Face · Krit ลงทะเบียน 4 session (s1–s2 ไม่ใส่แว่น, s3–s4 ใส่แว่น) · \"LFW คนอื่น\" = ทายเป็นคนใน LFW", { x: MX, y: 6.78, w: W - 2 * MX, h: 0.25, fontSize: 9.5, color: HEX.muted });
}

(async () => {
  await pres.writeFile({ fileName: OUT });
  if (process.env.SKILL) {
    const { applyTheme } = require(process.env.SKILL + "/scripts/apply_theme.js");
    await applyTheme(OUT, THEME);
  }
  console.log("wrote", OUT, pres.slides.length, "slides");
})();
