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

const VARS = ["full", "upper_half", "lower_half", "eyes_only", "nose_mouth", "mask", "sunglasses"];
const VTH = { full: "เต็ม", upper_half: "ครึ่งบน", lower_half: "ครึ่งล่าง", eyes_only: "ตา", nose_mouth: "จมูก+ปาก",
  mask: "หน้ากาก", sunglasses: "แว่นกันแดด" };
const METH = ["M0", "M1", "M2", "M3", "M4"];
const MNAME = { M0: "Eigenfaces + SVM", M1: "VGG-Face · เทรนภาพเต็ม", M2: "VGG-Face · เต็ม+บางส่วน",
  M3: "FaceNet · เทรนภาพเต็ม", M4: "FaceNet · เต็ม+บางส่วน" };

// ---------- theme ----------
const THEME = {
  name: "Partial Face",
  headFontFace: "Leelawadee UI",
  bodyFontFace: "Leelawadee UI",
  colors: { dk1: "1F2430", lt1: "FFFFFF", dk2: "141A26", lt2: "F4F6F9", accent1: "2A78D6", accent2: "D95926",
    accent3: "2F9E44", accent4: "D03B3B", accent5: "9CC2EE", accent6: "F4B08F", hlink: "2A78D6", folHlink: "6B6F76" },
};
const HEX = { ink: "1F2430", muted: "6B6F76", blue: "2A78D6", blueSoft: "E8F1FC", orange: "D95926",
  green: "2F9E44", red: "D03B3B", line: "DDE1E7", card: "FFFFFF", bg: "F4F6F9", grey: "A3A39E", dark: "141A26" };

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.title = "Face Recognition from Partial Face — Final";
const C = pres.SchemeColor;
const W = 13.333, H = 7.5, MX = 0.6;
const FOOT = "FACE RECOGNITION FROM PARTIAL FACE · FINAL PROJECT";

pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: HEX.bg },
  objects: [
    { text: { text: FOOT, options: { x: MX, y: 7.02, w: 7, h: 0.3, fontSize: 9, color: HEX.muted, margin: 0, charSpacing: 1 } } },
    { placeholder: { options: { name: "kicker", type: "body", x: MX, y: 0.38, w: 9, h: 0.32, fontSize: 11, color: HEX.blue,
      bold: true, margin: 0, charSpacing: 1, align: "left" }, text: "" } },
    { placeholder: { options: { name: "title", type: "title", x: MX, y: 0.68, w: W - 2 * MX, h: 0.75, fontSize: 28,
      bold: true, color: HEX.ink, margin: 0, valign: "middle", align: "left" }, text: "" } },
  ],
  slideNumber: { x: W - MX - 0.8, y: 7.02, w: 0.8, h: 0.3, fontSize: 9, color: HEX.muted, align: "right" },
});
pres.defineSlideMaster({
  title: "DARK",
  background: { color: HEX.dark },
  objects: [
    { text: { text: FOOT, options: { x: MX, y: 7.02, w: 7, h: 0.3, fontSize: 9, color: "8A93A6", margin: 0, charSpacing: 1 } } },
  ],
  slideNumber: { x: W - MX - 0.8, y: 7.02, w: 0.8, h: 0.3, fontSize: 9, color: "8A93A6", align: "right" },
});

// ---------- helpers ----------
function pngSize(p) {
  const b = fs.readFileSync(p);
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) };
}
function img(s, p, x, y, w, h, align = "center") {
  if (!has(p)) { s.addText(`missing: ${path.basename(p)}`, { x, y, w, h, isTextBox: true, color: HEX.red, fontSize: 12 }); return; }
  const z = pngSize(p);
  const r = Math.min(w / z.w, h / z.h);
  const iw = z.w * r, ih = z.h * r;
  const ix = align === "left" ? x : x + (w - iw) / 2;
  s.addImage({ path: p, x: ix, y: y + (h - ih) / 2, w: iw, h: ih, altText: path.basename(p) });
  return { x: ix, y: y + (h - ih) / 2, w: iw, h: ih };
}
function card(s, x, y, w, h, fill = HEX.card, line = HEX.line) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, fill: { color: fill }, line: { color: line, width: 0.75 }, rectRadius: 0.08 });
}
function text(s, t, o) { s.addText(t, { isTextBox: true, margin: 0, color: HEX.ink, fontSize: 14, valign: "top", ...o }); }
function content(kicker, title, accentPart) {
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: currentSection });
  s.addText(kicker, { placeholder: "kicker" });
  s.addText(accentPart ? [{ text: title }, { text: accentPart, options: { color: HEX.blue } }] : title, { placeholder: "title" });
  return s;
}
function tag(s, t, x, y, color = HEX.blue, w) {
  const ww = w || 0.25 + 0.095 * t.length;
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w: ww, h: 0.3, fill: { color }, line: { color }, rectRadius: 0.06 });
  s.addText(t, { x, y, w: ww, h: 0.3, isTextBox: true, margin: 0, fontSize: 10.5, bold: true, color: "FFFFFF", align: "center", valign: "middle" });
}
function stat(s, big, label, x, y, w, color = HEX.blue) {
  text(s, big, { x, y, w, h: 0.75, fontSize: 36, bold: true, color });
  text(s, label, { x, y: y + 0.78, w, h: 0.6, fontSize: 12, color: HEX.muted });
}
function codeBlock(s, code, x, y, w, h, fs_ = 11.5) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, fill: { color: "1E2433" }, line: { color: "1E2433" }, rectRadius: 0.06 });
  const runs = [];
  code.split("\n").forEach((ln, i, arr) => {
    const isComment = ln.trim().startsWith("#");
    runs.push({ text: ln.length ? ln : " ", options: { color: isComment ? "8FA1B3" : "E6EAF0", breakLine: i < arr.length - 1 } });
  });
  s.addText(runs, { x: x + 0.2, y: y + 0.15, w: w - 0.4, h: h - 0.3, isTextBox: true, margin: 0, fontFace: "Consolas",
    fontSize: fs_, valign: "top", lineSpacingMultiple: 1.05 });
}
function notesRow(s, items, x, y, w, rowH = 0.95, numColor = HEX.blue) {
  items.forEach(([head, body], i) => {
    const yy = y + i * rowH;
    s.addShape(pres.shapes.OVAL, { x, y: yy, w: 0.38, h: 0.38, fill: { color: numColor }, line: { color: numColor } });
    text(s, String(i + 1), { x, y: yy, w: 0.38, h: 0.38, fontSize: 12, bold: true, color: "FFFFFF", align: "center", valign: "middle" });
    text(s, head, { x: x + 0.55, y: yy - 0.02, w: w - 0.55, h: 0.32, fontSize: 14, bold: true });
    text(s, body, { x: x + 0.55, y: yy + 0.3, w: w - 0.55, h: rowH - 0.32, fontSize: 12, color: HEX.muted });
  });
}
let currentSection = "";
const PAGES_FILE = path.join(__dirname, "pages.json");
const PREV = has(PAGES_FILE) ? rd(PAGES_FILE) : {};
const PAGES = {};
function mark(key) { (PAGES[key] = PAGES[key] || []).push(pres.slides.length); }
function pg(...keys) {
  const n = keys.flatMap((k) => PREV[k] || []);
  if (!n.length) return "–";
  const u = [...new Set(n)].sort((a, b) => a - b), parts = [];
  for (let i = 0; i < u.length; i++) {
    let j = i;
    while (j + 1 < u.length && u[j + 1] === u[j] + 1) j++;
    parts.push(i === j ? `${u[i]}` : `${u[i]}–${u[j]}`);
    i = j;
  }
  return `หน้า ${parts.join(", ")}`;
}
function section(t) { currentSection = t; pres.addSection({ title: t }); }

// =====================================================================
section("Intro");
{ // 1 title
  const s = pres.addSlide({ masterName: "DARK", sectionTitle: currentSection });
  text(s, "COMPUTER VISION · FINAL PROJECT", { x: MX, y: 1.0, w: 8, h: 0.35, fontSize: 12, bold: true, color: "7FB0EE", charSpacing: 2 });
  text(s, [{ text: "Face Recognition", options: { breakLine: true } }, { text: "from Partial Face", options: { color: "7FB0EE" } }],
    { x: MX, y: 1.5, w: 7.2, h: 2.2, fontSize: 50, bold: true, color: "FFFFFF", fit: "none" });
  text(s, "การจดจำใบหน้าจากภาพที่ถูกบดบังบางส่วน ด้วย MTCNN, VGG-Face / FaceNet และ Transfer Learning",
    { x: MX, y: 3.85, w: 6.8, h: 0.9, fontSize: 17, color: "C9D3E3" });
  text(s, `ผลการทดลองจริง · LFW ${stats.n_classes} คน ${stats.n_images.toLocaleString()} ภาพ · FEI · ระบบวิดีโอ`,
    { x: MX, y: 5.0, w: 6.8, h: 0.4, fontSize: 13, color: "8A93A6" });
  // real faces strip on the right
  img(s, path.join(FIN, "example_all_variants.png"), 7.7, 1.2, 5.1, 1.2);
  img(s, path.join(FIN, "example_mtcnn.png"), 7.9, 2.65, 2.3, 2.3);
  img(s, path.join(FIN, "example_variant.png"), 10.35, 2.65, 2.3, 2.3);
  text(s, "ภาพจริงจากชุดทดสอบ: MTCNN → Align → หน้ากาก", { x: 7.9, y: 5.05, w: 4.8, h: 0.35, fontSize: 11, color: "8A93A6" });
}

{ // 2 feedback map
  const s = content("FEEDBACK → FINAL", "สิ่งที่ปรับจาก", " Progress Presentation");
  const rows = [
    ["แสดง confusion matrix ของการทดลองจริง ทุกการทดลอง", "Confusion matrix จริงครบ LFW 5 โมเดล × 7 รูปแบบ = 35 ชุด + FEI 35 ชุด + ขยาย M1 เทียบ M2", `${pg("cm")} · ภาคผนวก`],
    ["นำผลจริงของการปรับรูปมาแสดง: คน 1 คน ปิดหน้า / เห็นบางส่วน", "ภาพจริงของคนเดียวกันใน V0–V6 พร้อมพิกัดกล่องจากโค้ด", pg("person")],
    ["v0–v6 ชุดข้อมูลเป็นอย่างไร", "ตัวอย่างชุดทดสอบ V0–V6 จริง 8 คน + จำนวนภาพ Train / Test ทุกชุด", pg("vdata")],
    ["มีแต่ภาพแนวคิด ควรแสดงภาพจริงเพิ่ม", "ตัวอย่างจริง 1 ภาพผ่านทุกขั้น (ตัวเลขจริง) + ภาพที่โมเดลทายถูก/ผิด + วิดีโอ", pg("example", "real")],
    ["ไม่ได้อธิบายโค้ด", "อธิบายโค้ด 4 หน้า: โครงสร้าง, Align/Occlusion, Train/Split, Recognizer + วิดีโอ", pg("code")],
  ];
  const x0 = MX, y0 = 1.75, cw = [4.3, 6.0, 1.83];
  const hdr = ["ข้อเสนอแนะ", "สิ่งที่ทำใน Final", "ดูที่"];
  const tbl = [hdr.map((h) => ({ text: h, options: { bold: true, color: HEX.muted, fontSize: 12, fill: { color: HEX.bg } } }))];
  rows.forEach((r) => tbl.push([
    { text: r[0], options: { fontSize: 13.5, color: HEX.ink } },
    { text: r[1], options: { fontSize: 13.5, color: HEX.ink, bold: true } },
    { text: r[2], options: { fontSize: 12.5, color: HEX.blue, bold: true } }]));
  s.addTable(tbl, { x: x0, y: y0, w: cw.reduce((a, b) => a + b), colW: cw, rowH: [0.42, 0.85, 0.85, 0.85, 0.85, 0.85],
    border: { type: "solid", pt: 0.75, color: HEX.line }, fill: { color: HEX.card }, valign: "middle", margin: [4, 8, 4, 8] });
}

{ // 3 team
  const s = content("01 · TEAM", "สมาชิกในกลุ่ม", " และบทบาทความรับผิดชอบ");
  const team = [
    ["นายวรศักดิ์ แสนกล้า", "MODEL ARCHITECTURE", "Transfer Learning (VGG-Face, FaceNet) และชั้น Softmax"],
    ["นายธีรภัทร ศิริมา", "DATA ACQUISITION", "รวบรวมชุดข้อมูล LFW, FEI, Custom และคัดจำนวนภาพต่อคลาส"],
    ["นายรพีพงศ์ สุบรรณวิลาศ", "PREPROCESSING", "MTCNN, Alignment และการสร้างภาพใบหน้าบางส่วน V0–V6"],
    ["นายกฤต ไหลไผ่ทอง", "TESTING PIPELINE", "Pipeline ทดสอบ: ภาพ / เว็บแคม / วิดีโอ จนถึงทำนายผล"],
    ["นายอนุวัฒน์ เจริญสุข", "RESULT & CONFIDENCE", "วัดผล (Accuracy, F1, ROC, Confusion) และเกณฑ์ τ"],
  ];
  team.forEach((t, i) => {
    const x = MX + i * 2.45, y = 1.9;
    card(s, x, y, 2.27, 4.6);
    s.addShape(pres.shapes.OVAL, { x: x + 0.2, y: y + 0.25, w: 0.6, h: 0.6, fill: { color: HEX.blueSoft }, line: { color: HEX.blueSoft } });
    text(s, `0${i + 1}`, { x: x + 0.2, y: y + 0.25, w: 0.6, h: 0.6, align: "center", valign: "middle", bold: true, color: HEX.blue, fontSize: 15 });
    text(s, t[0], { x: x + 0.2, y: y + 1.05, w: 1.9, h: 0.9, fontSize: 15, bold: true });
    text(s, t[1], { x: x + 0.2, y: y + 2.0, w: 1.9, h: 0.5, fontSize: 10, bold: true, color: HEX.blue, charSpacing: 1 });
    text(s, t[2], { x: x + 0.2, y: y + 2.55, w: 1.9, h: 1.8, fontSize: 12.5, color: HEX.muted });
  });
}

{ // 4 problem + targets
  const s = content("02 · PROBLEM & TARGETS", "ใบหน้าถูกบดบัง", " = ระบบจดจำแม่นยำลดลง");
  text(s, "ระบบจดจำใบหน้าทั่วไปเทรนด้วยใบหน้าเต็ม แต่ในการใช้งานจริงผู้ใช้มักสวมหน้ากากอนามัย แว่นกันแดด หรือมีสิ่งบังบางส่วน ส่วนที่หายไปมักเป็นส่วนที่ใช้แยกบุคคล เช่น ดวงตา",
    { x: MX, y: 1.7, w: 6.3, h: 1.2, fontSize: 15 });
  card(s, MX, 3.0, 6.3, 1.45, HEX.blueSoft, HEX.blueSoft);
  text(s, [{ text: "โจทย์ · Closed-set identification", options: { bold: true, color: HEX.blue, breakLine: true } },
    { text: "ระบุว่าเป็นใครในกลุ่มคนที่รู้จัก จากภาพที่เห็นเพียงบางส่วน ให้แม่นยำใกล้เคียงใบหน้าเต็ม และตอบ Unknown เมื่อไม่มั่นใจ" }],
    { x: MX + 0.25, y: 3.1, w: 5.8, h: 1.25, fontSize: 13.5, valign: "middle" });
  const T = [["T1", "Accuracy ใบหน้าเต็ม (M2)", "≥ 95%"], ["T2", "Accuracy เมื่อสวมหน้ากาก / แว่น (M2)", "≥ 85%"],
    ["T3", "เทรนเพิ่มด้วยภาพบางส่วน ช่วยรูปแบบที่ยากที่สุดของ M1", "≥ +10 จุด"]];
  card(s, 7.25, 1.7, 5.48, 2.75);
  text(s, "เป้าหมายที่ตั้งไว้ก่อนทดลอง", { x: 7.5, y: 1.85, w: 4.8, h: 0.35, fontSize: 12.5, bold: true, color: HEX.muted });
  T.forEach((t, i) => {
    const y = 2.3 + i * 0.7;
    tag(s, t[0], 7.5, y + 0.08, HEX.blue, 0.55);
    text(s, t[1], { x: 8.2, y, w: 3.0, h: 0.6, fontSize: 13, valign: "middle" });
    text(s, t[2], { x: 11.0, y, w: 1.55, h: 0.6, fontSize: 21, bold: true, color: HEX.blue, align: "right", valign: "middle" });
  });
  img(s, path.join(FIN, "real_predictions_mask.png"), MX, 4.6, W - 2 * MX, 2.05);
  text(s, "ภาพจริงจากชุด Test ของ LFW, V5 หน้ากาก · เลือกแสดง 4 ภาพที่ M1 ผิดแต่ M2 ถูก, 2 ภาพที่ถูกทั้งคู่, 2 ภาพที่ M2 ผิด · กรอบเขียว/แดง = M2 ถูก/ผิด",
    { x: MX, y: 6.7, w: W - 2 * MX, h: 0.28, fontSize: 10.5, color: HEX.muted, align: "center" });
}

// =====================================================================
section("Data");
{ // 5 data sources with real images
  const s = content("03 · DATA", "ชุดข้อมูลที่ใช้", " และภาพตัวอย่างจริง");
  img(s, path.join(FIGD, "samples_per_class.png"), MX, 1.65, 6.6, 5.1);
  const x = 7.55;
  card(s, x, 1.65, 5.18, 1.75);
  tag(s, "การทดลองหลัก", x + 0.25, 1.85, HEX.blue, 1.35);
  text(s, "LFW · Labeled Faces in the Wild", { x: x + 0.25, y: 2.25, w: 4.7, h: 0.4, fontSize: 15, bold: true });
  text(s, "13,233 ภาพ · 5,749 คน · ภาพข่าวจริง แสง มุม สีหน้าไม่ควบคุม · kaggle.com/datasets/jessicali9530/lfw-dataset",
    { x: x + 0.25, y: 2.65, w: 4.7, h: 0.7, fontSize: 11.5, color: HEX.muted });
  card(s, x, 3.55, 5.18, 1.55);
  tag(s, "มุมหน้า", x + 0.25, 3.75, HEX.orange, 0.9);
  text(s, "FEI Face Database", { x: x + 0.25, y: 4.12, w: 4.7, h: 0.4, fontSize: 15, bold: true });
  text(s, feiStats ? `200 คน × 14 ภาพ (หมุนหน้า ~180°) · MTCNN พบหน้า ${feiStats.n_images.toLocaleString()} ภาพ · kaggle.com/datasets/whizzkid/fei-face-data`
    : "200 คน × 14 ภาพ · หมุนหน้า ~180° · kaggle.com/datasets/whizzkid/fei-face-data",
    { x: x + 0.25, y: 4.5, w: 4.7, h: 0.55, fontSize: 11.5, color: HEX.muted });
  card(s, x, 5.25, 5.18, 1.5);
  tag(s, "วิดีโอ", x + 0.25, 5.45, HEX.green, 0.75);
  text(s, "Custom: สมาชิก 5 คน", { x: x + 0.25, y: 5.82, w: 4.7, h: 0.4, fontSize: 15, bold: true });
  text(s, "ภาพ / คลิปจากเว็บแคมหรือมือถือ ≥ 2 session ต่อคน · ใช้ทดสอบระบบวิดีโอ", { x: x + 0.25, y: 6.2, w: 4.7, h: 0.5, fontSize: 11.5, color: HEX.muted });
}

{ // 6 subset numbers
  const s = content("03 · DATA — SUBSET", "คัดข้อมูลให้สมดุล", "ก่อนเทรน");
  const steps = [["LFW ทั้งหมด", "13,233", "ภาพ · 5,749 คน ส่วนใหญ่มีภาพเดียว"],
    ["คนที่มี ≥ 20 ภาพ", String(stats.n_classes), "คน · ให้ Test มีอย่างน้อย 6 ภาพ/คน"],
    ["จำกัด ≤ 50 ภาพ/คน", stats.n_images.toLocaleString(), "ภาพ · สุ่มด้วย seed 42 กันคลาสใหญ่ครอบงำ"],
    ["Stratified 70 / 30", `${NUM.train_images.toLocaleString()} / ${NUM.test_images}`, "ภาพ Train / Test · ทุกคนอยู่ทั้งสองชุด"]];
  steps.forEach((t, i) => {
    const x = MX + i * 3.08;
    card(s, x, 1.7, 2.9, 2.1);
    text(s, `STEP ${i + 1}`, { x: x + 0.25, y: 1.85, w: 2.4, h: 0.3, fontSize: 10, bold: true, color: HEX.blue });
    text(s, t[0], { x: x + 0.25, y: 2.15, w: 2.5, h: 0.35, fontSize: 13.5, bold: true });
    text(s, t[1], { x: x + 0.25, y: 2.55, w: 2.5, h: 0.65, fontSize: 30, bold: true, color: i === 3 ? HEX.blue : HEX.ink });
    text(s, t[2], { x: x + 0.25, y: 3.2, w: 2.5, h: 0.55, fontSize: 11, color: HEX.muted });
  });
  img(s, path.join(FIGD, "class_distribution.png"), MX, 4.0, 8.2, 2.75, "left");
  card(s, 9.1, 4.05, 3.63, 2.65, HEX.blueSoft, HEX.blueSoft);
  text(s, "MTCNN ตรวจพบหน้า", { x: 9.35, y: 4.25, w: 3.2, h: 0.35, fontSize: 13, bold: true });
  text(s, `${stats.mtcnn_detected.toLocaleString()} / ${stats.n_images.toLocaleString()}`, { x: 9.35, y: 4.6, w: 3.2, h: 0.6, fontSize: 28, bold: true, color: HEX.blue });
  text(s, `ภาพที่หาไม่เจอใช้ center crop แทน (${stats.mtcnn_fallback} ภาพ) · เดาสุ่มถูกเพียง 1/${stats.n_classes} ≈ ${f1(100 / stats.n_classes)}%`,
    { x: 9.35, y: 5.25, w: 3.2, h: 1.3, fontSize: 12, color: HEX.muted });
}

// =====================================================================
section("Pipeline");
{ // 7 pipeline overview
  const s = content("04 · PIPELINE", "ขั้นตอนการทำงาน", " ตั้งแต่ภาพถึงชื่อคน");
  const st = [["เลือกข้อมูล", "step1", "≥20, ≤50 ภาพ/คน"], ["MTCNN", "step1", "กรอบ + 5 จุด"], ["Alignment", "step1", "224×224"],
    ["Partial Face", "step2", "V0–V6 (7 แบบ)"], ["Feature", "step2", "VGG-Face fc7 4,096-d\nFaceNet 512-d"], ["Softmax", "step3", "ชื่อ + p, p<τ → Unknown"]];
  const imgs = ["example_raw.png", "example_mtcnn.png", "example_aligned.png", "example_variant.png", "example_features.png", "example_top5_M2.png"];
  st.forEach((t, i) => {
    const x = MX + i * 2.05, y = 1.75;
    card(s, x, y, 1.9, 4.2);
    text(s, `0${i + 1}`, { x: x + 0.18, y: y + 0.15, w: 0.6, h: 0.3, fontSize: 11, bold: true, color: HEX.blue });
    text(s, t[1] + ".py", { x: x + 0.8, y: y + 0.15, w: 0.95, h: 0.3, fontSize: 9.5, color: HEX.muted, align: "right", fontFace: "Consolas" });
    text(s, t[0], { x: x + 0.18, y: y + 0.45, w: 1.6, h: 0.4, fontSize: 15, bold: true });
    img(s, path.join(FIN, imgs[i]), x + 0.12, y + 0.95, 1.66, 1.9);
    text(s, t[2], { x: x + 0.18, y: y + 3.0, w: 1.6, h: 1.1, fontSize: 11.5, color: HEX.muted });
    if (i < 5) s.addShape(pres.shapes.CHEVRON, { x: x + 1.92, y: y + 1.8, w: 0.11, h: 0.22, fill: { color: HEX.blue }, line: { color: HEX.blue } });
  });
  card(s, MX, 6.1, W - 2 * MX, 0.7, HEX.blueSoft, HEX.blueSoft);
  text(s, "ภาพทุกช่องเป็นผลจริงของภาพทดสอบภาพเดียวกัน (ไม่ใช่ภาพแนวคิด) · เครื่องมือ: Python · PyTorch · facenet-pytorch · keras-vggface weights · scikit-learn · OpenCV",
    { x: MX + 0.25, y: 6.1, w: W - 2 * MX - 0.5, h: 0.7, fontSize: 12.5, valign: "middle" });
}

{ // 8 worked example 1
  const s = content("04 · WORKED EXAMPLE (ภาพจริง)", "ตัวอย่างจริง 1 ภาพ:", " ตรวจจับและจัดตำแหน่ง");
  mark("example");
  const nm = EX.true.replace(/_/g, " ");
  const panels = [["① ภาพจาก LFW", "example_raw.png", `${EX.raw_shape[1]}×${EX.raw_shape[0]}×3 · ${path.basename(EX.file)}`],
    ["② MTCNN กรอบ + 5 จุด", "example_mtcnn.png", `bbox (${EX.mtcnn_box.join(", ")}) · p = ${f1(EX.mtcnn_prob, 3)}`],
    ["③ Align → 224×224", "example_aligned.png", `หมุน ${f1(-EX.align_rotation_deg)}° · ย่อ/ขยาย ×${f1(EX.align_scale, 2)}`]];
  panels.forEach((p, i) => {
    const x = MX + i * 2.95;
    card(s, x, 1.7, 2.8, 4.0);
    text(s, p[0], { x: x + 0.2, y: 1.85, w: 2.4, h: 0.35, fontSize: 13.5, bold: true });
    img(s, path.join(FIN, p[1]), x + 0.2, 2.3, 2.4, 2.4);
    text(s, p[2], { x: x + 0.2, y: 4.85, w: 2.4, h: 0.75, fontSize: 11, color: HEX.muted });
  });
  const lm = EX.landmarks, tp = EX.template;
  const lab = ["ตาซ้าย", "ตาขวา", "ปลายจมูก", "มุมปากซ้าย", "มุมปากขวา"];
  const tbl = [[{ text: "จุด", options: { bold: true, color: HEX.muted } }, { text: "MTCNN (ภาพจริง)", options: { bold: true, color: HEX.muted } },
    { text: "แม่แบบ 224²", options: { bold: true, color: HEX.muted } }]];
  lab.forEach((l, i) => tbl.push([l, `(${lm[i][0]}, ${lm[i][1]})`, `(${tp[i][0].toFixed(0)}, ${tp[i][1].toFixed(0)})`]));
  s.addTable(tbl, { x: 9.55, y: 1.7, w: 3.18, colW: [1.08, 1.15, 0.95], rowH: 0.4, fontSize: 11.5, color: HEX.ink,
    border: { type: "solid", pt: 0.5, color: HEX.line }, fill: { color: HEX.card }, margin: [2, 5, 2, 5] });
  text(s, `ตาเอียง ${f1(EX.eye_tilt_deg)}° ในภาพจริง → Similarity transform (cv2.estimateAffinePartial2D) หมุน ย่อ เลื่อน ให้ 5 จุดตรงแม่แบบ`,
    { x: 9.55, y: 4.3, w: 3.18, h: 1.4, fontSize: 11.5, color: HEX.muted });
  card(s, MX, 5.9, W - 2 * MX, 0.85, HEX.blueSoft, HEX.blueSoft);
  text(s, `คนจริง: ${nm} (อยู่ในชุด Test ไม่เคยใช้เทรน) · หลัง Align ทุกภาพมีตา จมูก ปาก อยู่ที่พิกัดเดียวกัน กล่องบดบัง V1–V6 จึงตกที่ส่วนเดียวกันของทุกคน`,
    { x: MX + 0.25, y: 5.9, w: W - 2 * MX - 0.5, h: 0.85, fontSize: 13, valign: "middle" });
}

{ // 9 worked example 2
  const s = content("04 · WORKED EXAMPLE (ภาพจริง)", "ภาพเดียวกัน:", " บดบัง → Feature → Softmax");
  mark("example");
  card(s, MX, 1.7, 2.75, 4.15);
  text(s, "④ V5 หน้ากาก", { x: MX + 0.2, y: 1.85, w: 2.4, h: 0.35, fontSize: 13.5, bold: true });
  img(s, path.join(FIN, "example_variant.png"), MX + 0.2, 2.3, 2.35, 2.35);
  text(s, "ปิดกล่อง x 24–200, y 116–224 ตั้งแต่จมูกลงไป", { x: MX + 0.2, y: 4.8, w: 2.4, h: 0.9, fontSize: 11, color: HEX.muted });
  card(s, 3.5, 1.7, 3.05, 4.15);
  text(s, "⑤ VGG-Face fc7 (แช่แข็ง)", { x: 3.7, y: 1.85, w: 2.7, h: 0.35, fontSize: 13.5, bold: true });
  img(s, path.join(FIN, "example_features.png"), 3.6, 2.3, 2.85, 1.5);
  text(s, `เวกเตอร์ ${EX.feature_dim.toLocaleString()} มิติ (ReLU จึงมีค่า 0 มาก) · 8 ค่าแรก:
[${EX.feature_first8.map((v) => v.toFixed(2)).join(", ")}]
ทำ L2-normalize ก่อนเข้า Softmax`,
    { x: 3.7, y: 3.95, w: 2.7, h: 1.8, fontSize: 11, color: HEX.muted });
  card(s, 6.7, 1.7, 6.03, 4.15);
  text(s, "⑥ Softmax Top-5 (ค่าจริงของภาพนี้)", { x: 6.9, y: 1.85, w: 5.6, h: 0.35, fontSize: 13.5, bold: true });
  img(s, path.join(FIN, "example_top5_M1.png"), 6.8, 2.25, 2.9, 2.1);
  img(s, path.join(FIN, "example_top5_M2.png"), 9.75, 2.25, 2.9, 2.1);
  const t1 = EX.top5.M1[0], t2 = EX.top5.M2[0];
  const ok1 = t1[0] === EX.true, ok2 = t2[0] === EX.true;
  text(s, [{ text: `M1 (เทรนภาพเต็ม): ${t1[0].replace(/_/g, " ")} p = ${f1(t1[1], 2)} ${ok1 ? "✓" : "✗ ผิดคน"}${t1[1] < 0.5 ? " → p < τ ตอบ Unknown" : ""}`, options: { color: ok1 ? HEX.green : HEX.red, bold: true, breakLine: true } },
    { text: `M2 (เต็ม+บางส่วน): ${t2[0].replace(/_/g, " ")} p = ${f1(t2[1], 2)} ${ok2 ? "✓" : "✗"}${t2[1] >= 0.5 ? " → p ≥ τ ตอบชื่อ" : " → p < τ ตอบ Unknown"}`, options: { color: ok2 ? HEX.green : HEX.red, bold: true } }],
    { x: 6.9, y: 4.5, w: 5.65, h: 1.2, fontSize: 12.5, valign: "middle" });
  card(s, MX, 6.0, W - 2 * MX, 0.75, HEX.blueSoft, HEX.blueSoft);
  text(s, "ภาพเดียวกัน feature เดียวกัน ต่างกันแค่ชั้น Softmax: M2 เคยเห็นภาพบดบังตอนเทรน จึงจำคนได้แม้ปากและจมูกหายไป · (ภาพนี้เลือกจากกรณีที่ M1 ผิดแต่ M2 ถูก เพื่ออธิบายความต่าง ภาพรวมดูที่ Accuracy และ confusion matrix)",
    { x: MX + 0.25, y: 6.0, w: W - 2 * MX - 0.5, h: 0.75, fontSize: 13, valign: "middle" });
}

{ // 10 variant rules
  const s = content("04 · PARTIAL FACE V0–V6", "กติกาการสร้าง 7 รูปแบบ", " (พิกัดจริงจากโค้ด)");
  mark("person");
  img(s, path.join(FIN, "variant_rules.png"), MX, 1.6, W - 2 * MX, 2.6);
  const tbl = [["", "ชื่อในโค้ด", "วิธี", "กล่อง (x0, y0, x1, y1) บน 224×224", "สิ่งที่เหลือให้เห็น"].map((h) => ({ text: h, options: { bold: true, color: HEX.muted, fill: { color: HEX.bg } } }))];
  const what = { full: "ทั้งใบหน้า (อ้างอิง)", upper_half: "หน้าผาก คิ้ว ตา", lower_half: "ปลายจมูก ปาก คาง", eyes_only: "แถบดวงตาและคิ้ว",
    nose_mouth: "จมูกและปาก", mask: "ตา คิ้ว หน้าผาก", sunglasses: "ทุกส่วนยกเว้นดวงตา" };
  Object.entries(NUM.per_variant).forEach(([k, v]) => {
    const [code, name] = k.split(" ");
    tbl.push([{ text: code, options: { bold: true, color: HEX.blue } }, { text: name, options: { fontFace: "Consolas" } },
      v.box ? (v.rule === "keep" ? "เหลือเฉพาะกล่อง" : "ปิดกล่องเป็นสีดำ") : "–", v.box ? `(${v.box.join(", ")})` : "–", what[name]]);
  });
  s.addTable(tbl, { x: MX, y: 4.3, w: W - 2 * MX, colW: [0.6, 1.9, 2.2, 3.8, 3.633], rowH: 0.3, fontSize: 11.5, color: HEX.ink,
    border: { type: "solid", pt: 0.5, color: HEX.line }, fill: { color: HEX.card }, margin: [1, 6, 1, 6] });
}

{ // 11 one person
  const s = content("04 · ภาพจริงของคน 1 คน", `${NUM.one_person.name.replace(/_/g, " ")}:`, " ภาพจริง 4 ภาพ × 7 รูปแบบ");
  mark("person");
  img(s, path.join(FIN, "one_person_variants.png"), MX, 1.6, 9.0, 5.2, "left");
  card(s, 9.85, 1.7, 2.88, 5.0);
  notesRow(s, [["ภาพคนเดียวกัน", "ต่างแสง ต่างมุม ต่างวัน จาก LFW"], ["Align ก่อนบัง", "กล่องตกที่ตา/ปากตรงกันทุกภาพ"],
    ["ทุกภาพมี 7 ชุด", "V0 ใช้เทรน M1 · V0–V6 ใช้เทรน M2"], ["Test แยกทีละ V", "วัด Accuracy แยกทั้ง 7 รูปแบบ"]], 10.05, 1.95, 2.55, 1.18);
}

{ // 12 datasets V0-V6
  const s = content("04 · ชุดข้อมูล V0–V6", "ชุดข้อมูล V0–V6", " เป็นอย่างไร");
  mark("vdata");
  img(s, path.join(FIN, "dataset_variants.png"), MX, 1.55, 6.0, 5.3, "left");
  const n_tr = NUM.train_images, n_te = NUM.test_images;
  const tbl = [["ชุด", "Train", "Test"].map((h) => ({ text: h, options: { bold: true, color: HEX.muted, fill: { color: HEX.bg } } }))];
  VARS.forEach((v, i) => tbl.push([`V${i} ${VTH[v]}`, n_tr.toLocaleString(), n_te.toLocaleString()]));
  s.addTable(tbl, { x: 6.95, y: 1.7, w: 2.75, colW: [1.35, 0.7, 0.7], rowH: 0.36, fontSize: 11.5, color: HEX.ink,
    border: { type: "solid", pt: 0.5, color: HEX.line }, fill: { color: HEX.card }, margin: [1, 5, 1, 5], align: "left" });
  card(s, 9.95, 1.7, 2.78, 2.85, HEX.blueSoft, HEX.blueSoft);
  text(s, "ขนาดชุดเทรน", { x: 10.15, y: 1.85, w: 2.4, h: 0.35, fontSize: 13, bold: true });
  text(s, [{ text: "M1, M3 ", options: { bold: true } }, { text: `V0 อย่างเดียว\n${n_tr.toLocaleString()} ภาพ`, options: { breakLine: true } },
    { text: " ", options: { breakLine: true, fontSize: 6 } },
    { text: "M2, M4 ", options: { bold: true } }, { text: `V0–V6\n${(7 * n_tr).toLocaleString()} ภาพ (7 เท่า)`, options: {} }],
    { x: 10.15, y: 2.3, w: 2.45, h: 2.1, fontSize: 12.5 });
  card(s, 6.95, 4.8, 5.78, 1.95);
  text(s, `ทุกชุดใช้ภาพต้นฉบับชุดเดียวกัน (${stats.n_images.toLocaleString()} ภาพ, ${stats.n_classes} คน) แต่ปิด/เหลือคนละส่วน · ภาพ Test ${n_te} ภาพเดิมถูกทดสอบ 7 ครั้ง ครั้งละ 1 รูปแบบ · ภาพบดบังของภาพ Test ไม่เคยถูกใช้เทรน (ไม่มีข้อมูลรั่ว)`,
    { x: 7.15, y: 4.95, w: 5.4, h: 1.7, fontSize: 12.5 });
}

{ // 13 model & training
  const s = content("05 · MODEL & TRAINING", "Transfer Learning:", " เทรนเฉพาะชั้นสุดท้าย");
  const tbl = [["โมเดล", "Feature (แช่แข็ง)", "เทรนด้วย", "หมายเหตุ"].map((h) => ({ text: h, options: { bold: true, color: HEX.muted, fill: { color: HEX.bg } } }))];
  const rows = [["M0", "Eigenfaces (PCA 150) 64×64 เทา", "V0", "Baseline แบบดั้งเดิม + SVM"], ["M1", "VGG-Face fc7 · 4,096-d", "V0", "แบบ (ก)"],
    ["M2", "VGG-Face fc7 · 4,096-d", "V0–V6", "แบบ (ข) ใช้ตรวจ T1–T3"], ["M3", "FaceNet (VGGFace2) · 512-d", "V0", "backbone เปรียบเทียบ"],
    ["M4", "FaceNet (VGGFace2) · 512-d", "V0–V6", "backbone เปรียบเทียบ"]];
  rows.forEach((r) => tbl.push([{ text: r[0], options: { bold: true, color: HEX.blue } }, r[1], r[2], r[3]]));
  s.addTable(tbl, { x: MX, y: 1.75, w: 7.4, colW: [0.8, 2.75, 1.0, 2.85], rowH: 0.5, fontSize: 12.5, color: HEX.ink,
    border: { type: "solid", pt: 0.5, color: HEX.line }, fill: { color: HEX.card }, margin: [2, 6, 2, 6] });
  card(s, MX, 5.0, 7.4, 1.75, HEX.blueSoft, HEX.blueSoft);
  text(s, `Softmax = LogisticRegression (multinomial, C = 10) · VGG-Face: 4,096 × ${stats.n_classes} + ${stats.n_classes} = ${(4096 * stats.n_classes + stats.n_classes).toLocaleString()} พารามิเตอร์ที่เทรน · ทดสอบชุด Test เดียวกันทุกโมเดล แยกทีละ V0–V6 · seed 42`,
    { x: MX + 0.25, y: 5.1, w: 6.9, h: 1.55, fontSize: 13, valign: "middle" });
  const blocks = [["Input 224×224×3", "ภาพที่ Align แล้ว + บดบัง", HEX.card], ["conv1 – conv5", "13 ชั้น Convolution · FROZEN", HEX.card],
    ["fc6 → fc7", "4,096 → 4,096 · FROZEN", HEX.card], [`Softmax ${stats.n_classes} คลาส`, "ชั้นใหม่ · TRAIN", HEX.blueSoft]];
  blocks.forEach((b, i) => {
    const y = 1.75 + i * 1.27;
    card(s, 8.4, y, 4.33, 1.0, b[2], b[2] === HEX.card ? HEX.line : HEX.blue);
    text(s, b[0], { x: 8.65, y: y + 0.12, w: 3.9, h: 0.4, fontSize: 14, bold: true });
    text(s, b[1], { x: 8.65, y: y + 0.52, w: 3.9, h: 0.4, fontSize: 11.5, color: i === 3 ? HEX.blue : HEX.muted, bold: i === 3 });
  });
}

// =====================================================================
section("Code");
{ // 14 code map
  const s = content("06 · CODE", "โครงสร้างโค้ด:", " 1 ไฟล์ = 1 ขั้นตอน");
  mark("code");
  const files = [["common.py", "ค่าตั้งต้น, แม่แบบ 5 จุด, align(), VARIANTS + apply_variant(), augment()"],
    ["step1_detect_align.py", "เลือกคน (≥20, ≤50) → MTCNN → align → faces.npz + คนแปลกหน้า unknown.npz"],
    ["step2_extract_features.py", "สร้าง V0–V6 ของทุกภาพ → VGG-Face / FaceNet → features_*.npz"],
    ["step3_train_evaluate.py", "แบ่ง 70/30 → เทรน M0–M4 → ทดสอบ 7 รูปแบบ → metrics, targets, models/*.joblib"],
    ["step4 / step5 _figures.py", "กราฟ, ROC, confusion matrix ทุกการทดลอง, ภาพตัวอย่างจริง"],
    ["recognizer.py", "ใช้งานจริง: detect → align → feature → Softmax → ชื่อ หรือ Unknown"],
    ["predict.py · webcam.py · video.py · app.py", "ภาพนิ่ง · เว็บแคม · ไฟล์วิดีโอ (มีลูกศรชี้ชื่อ) · เว็บเดโม"]];
  files.forEach((f, i) => {
    const y = 1.75 + i * 0.71;
    card(s, MX, y, W - 2 * MX, 0.6);
    text(s, f[0], { x: MX + 0.25, y, w: 4.6, h: 0.6, fontSize: 13, bold: true, fontFace: "Consolas", valign: "middle", color: i >= 5 ? HEX.green : HEX.blue });
    text(s, f[1], { x: MX + 4.9, y, w: 7.0, h: 0.6, fontSize: 13, valign: "middle" });
  });
}

{ // 15 code: align + variants
  const s = content("06 · CODE — PREPROCESSING", "align() และ apply_variant()", " ใน common.py");
  mark("code");
  codeBlock(s, `TEMPLATE = np.array([[82.9, 94.7], [139.2, 94.4],   # ตาซ้าย, ตาขวา
                     [112.0, 126.8],                 # ปลายจมูก
                     [88.9, 165.4], [135.6, 165.1]]) # มุมปาก

def align(img, pts):
    M, _ = cv2.estimateAffinePartial2D(pts, TEMPLATE,
                                       method=cv2.LMEDS)
    return cv2.warpAffine(img, M, (224, 224))

VARIANTS = {"full": ("keep", None),
    "upper_half": ("keep", (0, 0, 224, 124)),
    "mask":       ("hide", (24, 116, 200, 224)), ...}

def apply_variant(img, name):
    mode, box = VARIANTS[name]
    if box is None: return img.copy()
    x0, y0, x1, y1 = box
    if mode == "keep":
        out = np.zeros_like(img)
        out[y0:y1, x0:x1] = img[y0:y1, x0:x1]
    else:
        out = img.copy(); out[y0:y1, x0:x1] = 0
    return out`, MX, 1.65, 7.3, 5.15, 11.5);
  notesRow(s, [["แม่แบบ 5 จุด", "ตำแหน่งที่ต้องการให้ ตา จมูก ปาก ไปอยู่ในภาพ 224×224"],
    ["Similarity transform", "หา M (หมุน + ย่อ/ขยาย + เลื่อน) จาก 5 จุดของ MTCNN ด้วย LMEDS ทนจุดผิด 1–2 จุด"],
    ["ตารางรูปแบบ", "แต่ละรูปแบบ = (keep หรือ hide, กล่อง) แก้ที่เดียวใช้ได้ทั้งเทรน ทดสอบ และเดโม"],
    ["keep", "สร้างภาพดำ แล้วคัดลอกเฉพาะกล่องกลับมา (V1–V4)"],
    ["hide", "คัดลอกทั้งภาพ แล้วถมกล่องเป็น 0 (V5 หน้ากาก, V6 แว่น)"]], 8.2, 1.75, 4.5, 1.0);
}

{ // 16 code: split + train
  const s = content("06 · CODE — TRAINING", "make_split() และ fit_deep()", " ใน step3");
  mark("code");
  codeBlock(s, `# 70/30 แบ่งตาม "ภาพ" แบบ stratified: ทุกคนอยู่ทั้ง 2 ชุด
idx_tr, idx_te = train_test_split(np.arange(len(y)),
                 test_size=0.3, stratify=y, random_state=42)

DEEP = {"M1": ("vggface", ["full"]),
        "M2": ("vggface", list(VARIANTS)),   # V0-V6
        "M3": ("facenet", ["full"]),
        "M4": ("facenet", list(VARIANTS))}

def fit_deep(m, feats, aug, y, idx):
    bb, used = DEEP[m]
    X = [feats[bb][v][idx] for v in used]      # เฉพาะภาพ train
    return LogisticRegression(C=10, max_iter=3000).fit(
        np.concatenate(X), np.tile(y[idx], len(X)))

for v in VARIANTS:                              # ทดสอบทีละรูปแบบ
    p = clf.predict_proba(feats[bb][v][idx_te])
    results[(m, v)] = metrics(y[idx_te], p, clf.classes_)`, MX, 1.65, 7.3, 5.15, 11.5);
  notesRow(s, [["แบ่งก่อนสร้างชุดเทรน", "ภาพ test และภาพบดบังของมันไม่เคยถูกใช้เทรน"],
    ["M1 กับ M2 ต่างกันที่เดียว", "รายการรูปแบบที่ใช้เทรน: [V0] หรือ [V0…V6]"],
    ["np.tile", "ภาพเดียวกัน 7 รูปแบบ ได้ label คนเดิม 7 ครั้ง"],
    ["Softmax", "LogisticRegression multinomial บน feature ที่ L2-normalize"],
    ["วัดผลแยกรูปแบบ", "Accuracy, P/R/F1, ROC-AUC, Top-5 และ confusion matrix ต่อ (โมเดล, V)"]], 8.2, 1.75, 4.5, 1.0);
}

{ // 17 code: recognizer
  const s = content("06 · CODE — INFERENCE", "Recognizer.__call__()", " ใช้กับภาพ เว็บแคม และวิดีโอ");
  mark("code");
  codeBlock(s, `def __call__(self, rgb, variant="full", tau=None):
    boxes, det, lms = self.mtcnn.detect(rgb, landmarks=True)
    keep = det >= self.min_detect            # ตัดกรอบหลอก
    faces = np.stack([apply_variant(align(rgb, p), variant)
                      for p in lms[keep]])
    feat = normalize(extract(self.model, self.prep, faces))
    prob = self.clf.predict_proba(feat)
    for box, p in zip(boxes, prob):
        best = np.argsort(-p)[:5]
        name = names[best[0]] if p[best[0]] >= tau else "Unknown"
        out.append(Face(box, ..., name, p[best[0]], top5))
    return out`, MX, 1.65, 7.3, 3.9, 11.5);
  codeBlock(s, `# video.py: คนเดียวกันอยู่หลายเฟรม -> โหวตชื่อในหน้าต่าง 15 เฟรม
for t, f in tracker.step(faces):            # จับคู่ด้วย IoU ของกรอบ
    t.update(f.box, f.top[0][0], f.prob)
    name, p = t.label(tau)                  # ผลรวม p สูงสุด
draw_callout(frame, t.box, f"{name} {p:.0%}", colour)  # กรอบ+ลูกศร`, MX, 5.7, 7.3, 1.1, 10.5);
  notesRow(s, [["ตรวจจับทุกหน้าในภาพ", "MTCNN ให้กรอบ ความมั่นใจ และ 5 จุด"], ["ขั้นเดียวกับตอนเทรน", "align → (บดบัง) → feature → normalize"],
    ["Top-5 + τ", "ถ้า p สูงสุด < τ ตอบ Unknown แทนการเดาชื่อ"], ["วิดีโอ", "ติดตามหน้าข้ามเฟรม + โหวต ลดชื่อกระพริบ"]], 8.2, 1.75, 4.5, 1.15);
}

// =====================================================================
section("Results");
{ // 18 accuracy table + chart
  const s = content("07 · RESULTS", "ผลการทดลองจริง:", " 5 โมเดล × 7 รูปแบบ");
  const tbl = [[{ text: "Accuracy (%)", options: { bold: true, color: HEX.muted, fill: { color: HEX.bg } } },
    ...METH.map((m) => ({ text: m, options: { bold: true, color: m === "M2" ? HEX.blue : HEX.muted, fill: { color: HEX.bg }, align: "center" } }))]];
  VARS.forEach((v, i) => {
    const best = Math.max(...METH.map((m) => acc(m, v)));
    tbl.push([`V${i} ${VTH[v]}`, ...METH.map((m) => ({ text: f1(acc(m, v)),
      options: { align: "center", bold: acc(m, v) === best, color: acc(m, v) === best ? HEX.blue : HEX.ink } }))]);
  });
  s.addTable(tbl, { x: MX, y: 1.7, w: 5.3, colW: [1.6, 0.74, 0.74, 0.74, 0.74, 0.74], rowH: 0.42, fontSize: 12.5, color: HEX.ink,
    border: { type: "solid", pt: 0.5, color: HEX.line }, fill: { color: HEX.card }, margin: [2, 5, 2, 5] });
  text(s, METH.map((m) => `${m} ${MNAME[m]}`).join("\n"), { x: MX, y: 5.2, w: 5.3, h: 1.55, fontSize: 11, color: HEX.muted });
  const data = METH.map((m) => ({ name: `${m} ${MNAME[m]}`, labels: VARS.map((v, i) => `V${i} ${VTH[v]}`), values: VARS.map((v) => +f1(acc(m, v))) }));
  s.addChart(pres.charts.BAR, data, { x: 6.1, y: 1.6, w: 6.63, h: 5.2, barDir: "col", barGrouping: "clustered", barGapWidthPct: 40,
    chartColors: ["A3A39E", "9CC2EE", "2A78D6", "F4B08F", "D95926"], valAxisMaxVal: 100, valAxisMinVal: 0,
    valAxisLabelFontSize: 10, catAxisLabelFontSize: 10, valAxisLabelColor: HEX.muted, catAxisLabelColor: HEX.ink,
    catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt", legendFontFace: "+mn-lt", titleFontFace: "+mn-lt",
    valGridLine: { color: "E3E6EA", size: 0.5 }, catGridLine: { style: "none" }, showLegend: true, legendPos: "b", legendFontSize: 10,
    showTitle: true, title: "Accuracy (%) แยกตามรูปแบบที่ใช้ทดสอบ", titleFontSize: 12, titleColor: HEX.ink });
}

{ // 19 targets
  const s = content("07 · TARGETS", "ผลเทียบเป้าหมาย", " T1 · T2 · T3");
  const t1 = targets.T1, t2 = targets.T2, t3 = targets.T3;
  const hard = t3.M1_worst_variant;
  const cards = [
    ["T1", "Accuracy ใบหน้าเต็ม (M2)", `${f1(100 * t1.value)}%`, `เป้า ≥ 95% · ${t1.pass ? "ผ่าน" : "ไม่ผ่าน"}`, t1.pass],
    ["T2", "หน้ากาก / แว่นกันแดด (M2)", `${f1(100 * t2.value.mask)} / ${f1(100 * t2.value.sunglasses)}%`, `เป้า ≥ 85% · ${t2.pass ? "ผ่าน" : "ไม่ผ่าน"}`, t2.pass],
    ["T3", `รูปแบบที่ M1 แย่สุด: ${hard.map((v) => VTH[v]).join(", ")}`, hard.map((v) => `+${f1(100 * t3.value[v])}`).join(" / ") + " จุด",
      hard.map((v) => `${f1(acc("M1", v))} → ${f1(acc("M2", v))}%`).join(" · ") + ` · เป้า ≥ +10 · ${t3.pass ? "ผ่าน" : "ไม่ผ่าน"}`, t3.pass]];
  cards.forEach((c, i) => {
    const x = MX + i * 4.1;
    card(s, x, 1.75, 3.9, 2.9);
    tag(s, c[0], x + 0.25, 1.95, HEX.blue, 0.55);
    tag(s, c[4] ? "PASS" : "FAIL", x + 3.0, 1.95, c[4] ? HEX.green : HEX.red, 0.7);
    text(s, c[1], { x: x + 0.25, y: 2.4, w: 3.45, h: 0.6, fontSize: 13, color: HEX.muted });
    text(s, c[2], { x: x + 0.25, y: 3.0, w: 3.45, h: 0.8, fontSize: i === 2 ? 26 : 32, bold: true, color: HEX.blue, valign: "middle" });
    text(s, c[3], { x: x + 0.25, y: 3.85, w: 3.45, h: 0.7, fontSize: 11.5, color: HEX.muted });
  });
  const gains = VARS.slice(1).map((v) => [v, acc("M2", v) - acc("M1", v)]);
  s.addChart(pres.charts.BAR, [
    { name: "VGG-Face: M2 − M1", labels: VARS.slice(1).map((v) => VTH[v]), values: gains.map(([, g]) => +f1(g)) },
    { name: "FaceNet: M4 − M3", labels: VARS.slice(1).map((v) => VTH[v]), values: VARS.slice(1).map((v) => +f1(acc("M4", v) - acc("M3", v))) }],
    { x: MX, y: 4.8, w: 5.6, h: 2.05, barDir: "col", barGrouping: "clustered", chartColors: ["2A78D6", "D95926"],
      showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 9, dataLabelColor: HEX.ink, dataLabelFormatCode: "0",
      valAxisHidden: true, valGridLine: { style: "none" }, catGridLine: { style: "none" }, catAxisLabelFontSize: 10, catAxisLabelColor: HEX.ink,
      catAxisLabelFontFace: "+mn-lt", dataLabelFontFace: "+mn-lt", legendFontFace: "+mn-lt", titleFontFace: "+mn-lt",
      showLegend: true, legendPos: "r", legendFontSize: 10, showTitle: true, title: "เทรนเพิ่มด้วยภาพบางส่วน ได้เพิ่มกี่จุด", titleFontSize: 11, titleColor: HEX.ink });
  text(s, [{ text: "M2 − M1 ทุกรูปแบบที่ถูกบดบัง (จุด)", options: { bold: true, breakLine: true } },
    { text: gains.map(([v, g]) => `${VTH[v]} +${f1(g)}`).join(" · "), options: { color: HEX.blue, breakLine: true } },
    { text: `ROC-AUC (macro, OvR) ของ M2 ทุกรูปแบบ ≥ ${f1(Math.min(...VARS.map((v) => ACC[`M2|${v}`].auc)), 3)} · ข้อสังเกต: FaceNet เทรนเต็ม+บางส่วน (M4) ได้ ${f1(acc("M4", "sunglasses"))}% กับแว่น สูงกว่า M2`, options: { color: HEX.muted } }],
    { x: 6.55, y: 4.95, w: 6.18, h: 1.8, fontSize: 12.5 });
}

{ // 20 confusion grid
  const s = content("07 · CONFUSION MATRIX ทุกการทดลอง", "Confusion matrix จริง", ` ครบ 35 การทดลอง (${stats.n_classes}×${stats.n_classes})`);
  mark("cm");
  img(s, path.join(FIN, "confusion_grid.png"), MX, 1.55, 9.6, 5.3, "left");
  card(s, 10.4, 1.7, 2.33, 5.0);
  text(s, [{ text: "อ่านอย่างไร", options: { bold: true, breakLine: true } },
    { text: "แถว = คนจริง, คอลัมน์ = คนที่ทาย ทำ normalize ต่อแถว", options: { breakLine: true } },
    { text: " ", options: { breakLine: true, fontSize: 6 } },
    { text: "เส้นทแยงเข้ม = ทายถูก", options: { breakLine: true } },
    { text: "จุดนอกเส้นทแยง = สับสนคู่นั้น", options: { breakLine: true } },
    { text: " ", options: { breakLine: true, fontSize: 6 } },
    { text: "ตัวเลขบนช่อง = Accuracy", options: { breakLine: true } },
    { text: " ", options: { breakLine: true, fontSize: 6 } },
    { text: "M1/M3 พังที่ V2–V4, M2/M4 เส้นทแยงยังชัดทุกรูปแบบ · ภาพเต็มรายโมเดลอยู่ภาคผนวก", options: { color: HEX.blue, bold: true } }],
    { x: 10.6, y: 1.9, w: 1.95, h: 4.7, fontSize: 12, color: HEX.ink });
}

for (const [v, idx] of [["mask", 5], ["sunglasses", 6], ["lower_half", 2]]) { // 21-23 top-12
  const s = content("07 · CONFUSION MATRIX — M1 vs M2", `V${idx} ${VTH[v]}:`, " คนที่ M1 สับสนมากที่สุด 12 คน");
  mark("cm");
  img(s, path.join(FIN, `cm_top12_${v}_M1_vs_M2.png`), MX, 1.55, 9.4, 5.25, "left");
  card(s, 10.2, 1.7, 2.53, 5.0);
  const a1 = acc("M1", v), a2 = acc("M2", v);
  text(s, "Accuracy ทั้งชุด", { x: 10.4, y: 1.9, w: 2.1, h: 0.35, fontSize: 12, bold: true, color: HEX.muted });
  text(s, `${f1(a1)}%`, { x: 10.4, y: 2.25, w: 2.1, h: 0.6, fontSize: 26, bold: true, color: HEX.grey });
  text(s, "M1 เทรนภาพเต็ม", { x: 10.4, y: 2.85, w: 2.1, h: 0.3, fontSize: 11, color: HEX.muted });
  text(s, `${f1(a2)}%`, { x: 10.4, y: 3.3, w: 2.1, h: 0.6, fontSize: 26, bold: true, color: HEX.blue });
  text(s, "M2 เต็ม+บางส่วน", { x: 10.4, y: 3.9, w: 2.1, h: 0.3, fontSize: 11, color: HEX.muted });
  text(s, `ตัวเลข = จำนวนภาพ Test · สีแดง = ทายผิดเป็นคนนั้น · ช่องนอกเส้นทแยงของ M2 ลดลงชัดเจน`, { x: 10.4, y: 4.45, w: 2.15, h: 2.1, fontSize: 11.5 });
}

{ // 24 real predictions
  const s = content("07 · ตัวอย่างผลจริง", "ภาพจริงที่โมเดลทาย:", " ถูก / ผิด");
  mark("real");
  img(s, path.join(FIN, "real_predictions_sunglasses.png"), MX, 1.55, W - 2 * MX, 2.6);
  img(s, path.join(FIN, "real_predictions_lower_half.png"), MX, 4.2, W - 2 * MX, 2.6);
  text(s, "บน: V6 แว่นกันแดด · ล่าง: V2 ครึ่งล่าง · แต่ละแถวเลือก 4 ภาพที่ M1 ผิดแต่ M2 ถูก, 2 ภาพที่ถูกทั้งคู่, 2 ภาพที่ M2 ผิด · ✓ ✗ ใต้ภาพ = คำตอบจริงของ M1 / M2 พร้อม p",
    { x: MX, y: 6.8, w: W - 2 * MX, h: 0.25, fontSize: 10.5, color: HEX.muted });
}

{ // 25 open set / tau
  const s = content("07 · UNKNOWN & τ", "คนแปลกหน้า:", " เกณฑ์ความมั่นใจ τ");
  img(s, path.join(FIGD, "threshold_M2.png"), MX, 1.6, 6.0, 4.0, "left");
  img(s, path.join(FIGD, "open_set_M2.png"), 6.75, 1.6, 6.0, 4.0, "left");
  let msg = "ยังไม่มีผลชุดคนแปลกหน้า";
  if (openSet) {
    const rows = openSet.rows.filter((r) => r.method === "M2" && r.variant === "full");
    const at = rows.find((r) => Math.abs(r.tau - 0.5) < 1e-6);
    const ok = rows.find((r) => r.stranger_accepted <= 0.05);
    msg = `ทดสอบกับคนแปลกหน้า ${openSet.n_strangers} คนจาก LFW (ไม่เคยเทรน) · ที่ τ = 0.5 (ภาพเต็ม) คนแปลกหน้าถูกตอบเป็นชื่อ ${f1(100 * at.stranger_accepted)}% และคนที่รู้จักถูกเรียกชื่อถูก ${f1(100 * at.known_correct)}%`
      + (ok ? ` · τ = ${f1(ok.tau, 2)} ให้คนแปลกหน้าผ่าน ≤ 5% (รู้จักถูก ${f1(100 * ok.known_correct)}%)` : "");
  }
  card(s, MX, 5.75, W - 2 * MX, 1.0, HEX.blueSoft, HEX.blueSoft);
  text(s, msg, { x: MX + 0.25, y: 5.75, w: W - 2 * MX - 0.5, h: 1.0, fontSize: 12.5, valign: "middle" });
}

if (has(path.join(FIGD, "augmentation_effect.png"))) { // 26 augmentation
  const s = content("07 · DATA AUGMENTATION", "Data Augmentation:", " ช่วยแค่ไหน");
  img(s, path.join(FIGD, "augmentation.png"), MX, 1.6, 5.0, 2.4, "left");
  img(s, path.join(FIGD, "augmentation_effect.png"), 5.9, 1.6, 6.83, 5.15);
  const k = (ACC["M2+aug|full"] || {}).augmented_copies;
  const rows = ["M1", "M2", "M3", "M4"].filter((m) => ACC[`${m}+aug|full`]);
  const tbl = [["", "ไม่ aug", "+aug"].map((h) => ({ text: h, options: { bold: true, color: HEX.muted, fill: { color: HEX.bg } } }))];
  rows.forEach((m) => {
    const mean = (mm) => VARS.slice(1).reduce((a, v) => a + acc(mm, v), 0) / 6;
    tbl.push([`${m} เฉลี่ย V1–V6`, f1(mean(m)), { text: f1(mean(m + "+aug")), options: { bold: true, color: HEX.blue } }]);
  });
  s.addTable(tbl, { x: MX, y: 4.2, w: 5.0, colW: [2.4, 1.3, 1.3], rowH: 0.38, fontSize: 12, color: HEX.ink,
    border: { type: "solid", pt: 0.5, color: HEX.line }, fill: { color: HEX.card }, margin: [1, 6, 1, 6] });
  text(s, `สุ่มหมุน ±15°, ครอบ/ซูม, พลิก, ความสว่าง, noise · ${k || "K"} สำเนาต่อภาพ Train · ทดสอบกับภาพต้นฉบับเสมอ`,
    { x: MX, y: 6.2, w: 5.0, h: 0.6, fontSize: 11, color: HEX.muted });
}

if (feiM) { // 27 FEI
  const s = content("07 · FEI FACE DATABASE", "FEI:", " มุมหน้ากระทบผลอย่างไร");
  const fa = {}; for (const r of feiM) fa[`${r.method}|${r.variant}`] = 100 * r.accuracy;
  img(s, path.join(FEI, "figures", "pose_fei.png"), MX, 1.6, 7.2, 4.2, "left");
  const tbl = [["FEI Accuracy (%)", "M1", "M2", "M4"].map((h) => ({ text: h, options: { bold: true, color: HEX.muted, fill: { color: HEX.bg } } }))];
  VARS.forEach((v, i) => tbl.push([`V${i} ${VTH[v]}`, ...["M1", "M2", "M4"].map((m) => f1(fa[`${m}|${v}`]))]));
  s.addTable(tbl, { x: 8.1, y: 1.65, w: 4.63, colW: [1.9, 0.91, 0.91, 0.91], rowH: 0.38, fontSize: 12, color: HEX.ink,
    border: { type: "solid", pt: 0.5, color: HEX.line }, fill: { color: HEX.card }, margin: [1, 6, 1, 6], align: "left" });
  card(s, MX, 5.95, W - 2 * MX, 0.8, HEX.blueSoft, HEX.blueSoft);
  text(s, `${feiStats.n_classes} คน · MTCNN พบหน้า ${feiStats.n_images.toLocaleString()} ภาพ (ข้าม ${feiStats.skipped_no_face} ภาพ ส่วนใหญ่เป็นหน้าด้านข้าง) · แบ่ง 70/30 แบบเดียวกับ LFW · มุมหน้าด้านข้างเป็นจุดอ่อนหลักของ Pipeline นี้`,
    { x: MX + 0.25, y: 5.95, w: W - 2 * MX - 0.5, h: 0.8, fontSize: 12.5, valign: "middle" });
}

if (feiM && has(path.join(FEI, "figures", "final", "confusion_grid.png"))) { // FEI confusion grid
  const s = content("07 · CONFUSION MATRIX — FEI", "Confusion matrix จริงของ FEI", ` (${feiStats.n_classes}×${feiStats.n_classes})`);
  mark("cm");
  img(s, path.join(FEI, "figures", "final", "confusion_grid.png"), MX, 1.55, 9.6, 5.3, "left");
  card(s, 10.4, 1.7, 2.33, 5.0);
  text(s, [{ text: "FEI ภาพถ่ายในห้องควบคุม", options: { bold: true, breakLine: true } },
    { text: "พื้นหลังขาว แสงเดียวกัน แต่หันหน้าหลายมุม", options: { breakLine: true } },
    { text: " ", options: { breakLine: true, fontSize: 6 } },
    { text: "M1 เสียมากเมื่อเหลือแค่จมูก+ปาก หรือใส่แว่น", options: { breakLine: true } },
    { text: " ", options: { breakLine: true, fontSize: 6 } },
    { text: "M2 / M4 เส้นทแยงยังชัดทุกรูปแบบ เหมือนผลบน LFW", options: { color: HEX.blue, bold: true } }],
    { x: 10.6, y: 1.9, w: 1.95, h: 4.7, fontSize: 12, color: HEX.ink });
}

// =====================================================================
section("Video demo");
{ // 28 video design
  const s = content("08 · VIDEO DEMO", "ระบบวิดีโอ:", " ระบุว่าใครเป็นใคร พร้อมลูกศรชี้");
  const steps = [["อ่านเฟรม", "ไฟล์ mp4 หรือกล้อง"], ["MTCNN", "ทุกหน้าในเฟรม"], ["Align + Feature", "เหมือนตอนเทรน"], ["Softmax", "ชื่อ + p ต่อหน้า"],
    ["Tracking (IoU)", "จับคู่หน้าเดิมข้ามเฟรม"], ["โหวต 15 เฟรม", "ชื่อไม่กระพริบ, p<τ → Unknown"], ["วาดผล", "กรอบ + ลูกศร + ชื่อ + %"]];
  steps.forEach((t, i) => {
    const x = MX + i * 1.76;
    card(s, x, 1.7, 1.62, 1.35, i >= 4 ? HEX.blueSoft : HEX.card, i >= 4 ? HEX.blue : HEX.line);
    text(s, t[0], { x: x + 0.12, y: 1.8, w: 1.4, h: 0.45, fontSize: 12.5, bold: true });
    text(s, t[1], { x: x + 0.12, y: 2.27, w: 1.4, h: 0.75, fontSize: 10.5, color: HEX.muted });
  });
  const frame = path.join(VID, "frame_scene2.png");
  img(s, frame, MX, 3.25, 6.3, 3.55, "left");
  card(s, 7.15, 3.25, 5.58, 3.5);
  text(s, "Input ที่ต้องเตรียม (ตอบคำถาม: ต้องใช้ภาพอะไรบ้าง)", { x: 7.4, y: 3.42, w: 5.1, h: 0.4, fontSize: 13.5, bold: true });
  text(s, [{ text: "ภาพลงทะเบียนสมาชิกแต่ละคน 30–40 ภาพ ≥ 2 session (หน้าเต็ม ไม่ต้องบัง)", options: { bullet: true, breakLine: true } },
    { text: "คลิปทดสอบแยกต่างหาก: สมาชิกใส่หน้ากาก / แว่น / เอามือบัง", options: { bullet: true, breakLine: true } },
    { text: "คนนอกกลุ่ม 2–3 คน เพื่อทดสอบการตอบ Unknown", options: { bullet: true, breakLine: true } },
    { text: " ", options: { breakLine: true, fontSize: 6 } },
    { text: "ทดสอบระบบแล้วด้วยคลิปที่สร้างจากภาพ Test ของ LFW (ซ้าย): ใส่หน้ากาก แว่น มือบัง แบบวาดบนภาพจริง ไม่ใช่กล่องบน Align", options: { color: HEX.muted } }],
    { x: 7.4, y: 3.9, w: 5.1, h: 2.8, fontSize: 12.5, paraSpaceAfter: 4 });
}

if (video) { // 29 video results
  const s = content("08 · VIDEO DEMO — SMOKE TEST", "ผลทดสอบคลิป:", ` M2 ถูก ${video.correct}/${video.total} คน`);
  mark("real");
  img(s, path.join(VID, "frame_scene1.png"), MX, 1.6, 4.0, 2.25, "left");
  img(s, path.join(VID, "frame_scene3.png"), MX, 4.0, 4.0, 2.25, "left");
  const tbl = [["ฉาก", "ตำแหน่ง", "สิ่งบดบัง", "คนจริง", "M2 ตอบ (p)", "M1 ตอบ (p)"].map((h) => ({ text: h, options: { bold: true, color: HEX.muted, fill: { color: HEX.bg } } }))];
  const OCC = { none: "ไม่มี", mask: "หน้ากาก", glasses: "แว่นดำ", hand: "มือบังครึ่งหน้า" };
  const POS = { left: "ซ้าย", middle: "กลาง", right: "ขวา" };
  const cell = (r) => ({ text: `${r.shown.replace(/_/g, " ")} (${f1(r.p, 2)})`, options: { bold: true, color: r.shown === r.true ? HEX.green : HEX.red } });
  video.rows.forEach((r, i) => tbl.push([String(r.scene), POS[r.position], OCC[r.occluder], r.true.replace(/_/g, " "),
    cell(r), videoM1 ? cell(videoM1.rows[i]) : "–"]));
  s.addTable(tbl, { x: 4.85, y: 1.6, w: 7.88, colW: [0.5, 0.75, 1.25, 1.8, 1.79, 1.79], rowH: 0.42, fontSize: 10.5, color: HEX.ink,
    border: { type: "solid", pt: 0.5, color: HEX.line }, fill: { color: HEX.card }, margin: [1, 5, 1, 5] });
  text(s, `M2 ถูก ${video.correct}/${video.total} · M1 ถูก ${videoM1 ? videoM1.correct : "–"}/${video.total} (แต่ละโมเดลใช้ τ ที่ให้คนแปลกหน้าผ่าน ≤ 5%: M2 τ = ${f1(video.tau, 2)}, M1 τ = ${videoM1 ? f1(videoM1.tau, 2) : "–"}) · ทั้งสองเทรนบน Train split เท่านั้น ภาพในคลิปมาจากชุด Test · ${video.frames} เฟรม · ${f1(video.fps, 1)} เฟรม/วินาที บน ${video.device}`,
    { x: 4.85, y: 6.3, w: 7.88, h: 0.5, fontSize: 11, color: HEX.muted });
}

{ // 30 members pending
  const s = content("08 · CUSTOM DATASET (สมาชิกกลุ่ม)", "ขั้นตอนใช้กับสมาชิก 5 คน", " จากคลิปวิดีโอ");
  codeBlock(s, `# 1) ลงทะเบียน: คลิปหน้าเต็มของแต่ละคน 2 คลิป (คนละแสง/วัน)
python capture_faces.py --name Worasak --video worasak_1.mp4 --session s1 --count 40
python capture_faces.py --name Worasak --video worasak_2.mp4 --session s2 --count 40
python capture_faces.py --name Stranger1 --video friend.mp4 --stranger

# 2) เทรน (LFW + สมาชิก) แล้ววัดผล: confusion matrix 5 คน จาก session ที่ไม่เคยเห็น
python step1_detect_align.py --dataset lfw+custom
python step2_extract_features.py --dataset lfw+custom --augment 2
python step3_train_evaluate.py --dataset lfw+custom --fit-all
python step4_figures.py --dataset lfw+custom
python step5_final_figures.py --dataset lfw+custom

# 3) คลิปทดสอบ (ใส่หน้ากาก/แว่น) -> mp4 ที่มีกรอบ ลูกศร ชื่อ
python video.py group_test.mp4 --model "../models/lfw+custom_M2+aug.joblib"`, MX, 1.65, 8.1, 5.1, 11);
  card(s, 8.95, 1.65, 3.78, 5.1, "FFF4EC", HEX.orange);
  tag(s, "รอข้อมูลสมาชิก", 9.2, 1.9, HEX.orange, 1.55);
  text(s, [{ text: "ช่องนี้สำหรับผลจริงของกลุ่ม", options: { bold: true, breakLine: true } },
    { text: "Confusion matrix ของสมาชิก 5 คน + คอลัมน์ \"LFW คนอื่น\" (members_confusion_M2+aug.png ที่ results/lfw+custom/figures/final/)", options: { bullet: true, breakLine: true } },
    { text: "ภาพหน้าจอจากคลิป annotated.mp4", options: { bullet: true, breakLine: true } },
    { text: "อัตรา Unknown ของคนนอกกลุ่ม (video.py ใช้ τ ที่ให้คนแปลกหน้าผ่าน ≤ 5% อัตโนมัติ)", options: { bullet: true, breakLine: true } },
    { text: " ", options: { breakLine: true, fontSize: 6 } },
    { text: "ใช้ session ที่ 2 เป็นชุด Test อัตโนมัติ เพราะเฟรมจากคลิปเดียวกันเกือบซ้ำกัน", options: { color: HEX.muted } }],
    { x: 9.2, y: 2.4, w: 3.3, h: 4.2, fontSize: 12.5, paraSpaceAfter: 4 });
}

// =====================================================================
section("Wrap-up");
{ // 31 summary
  const s = content("SUMMARY", "สรุปผล", " และข้อจำกัด");
  const pts = [["ผลหลัก", `เทรนเพิ่มด้วยภาพบางส่วน (M2) ได้ ${f1(acc("M2", "full"))}% กับหน้าเต็ม และ ${f1(acc("M2", "mask"))} / ${f1(acc("M2", "sunglasses"))}% กับหน้ากาก / แว่น ผ่าน T1–T3 ทุกข้อ`],
    ["สิ่งที่ช่วยมากที่สุด", `ไม่ใช่ backbone ใหม่ แต่การให้ Softmax เห็นภาพบดบังตอนเทรน: ครึ่งล่าง ${f1(acc("M1", "lower_half"))} → ${f1(acc("M2", "lower_half"))}%`],
    ["Confusion matrix", "M1 สับสนกระจายทั่วเมื่อไม่เห็นดวงตา M2 / M4 เส้นทแยงยังชัดทุกรูปแบบ"],
    ["ข้อจำกัด", "MTCNN อาจหาหน้าไม่เจอเมื่อบังมากหรือหันข้าง (FEI) · Softmax ต้องเลือกคนที่รู้จักเสมอ จึงต้องตั้ง τ สำหรับคนแปลกหน้า · การบดบังในการทดลองเป็นกล่องบนภาพ Align"],
    ["ต่อยอด", "เก็บคลิปสมาชิกจริง 2 session · ทดสอบหน้ากากจริง · ใช้ detector ที่ทนการบังมากขึ้น"]];
  pts.forEach((p, i) => {
    const y = 1.7 + i * 1.02;
    card(s, MX, y, W - 2 * MX, 0.9);
    text(s, p[0], { x: MX + 0.3, y, w: 2.6, h: 0.9, fontSize: 15, bold: true, color: i === 3 ? HEX.orange : HEX.blue, valign: "middle" });
    text(s, p[1], { x: MX + 3.0, y, w: 8.9, h: 0.9, fontSize: 13, valign: "middle" });
  });
}

{ // 32 references
  const s = content("REFERENCES", "อ้างอิง", "");
  const refs = ["[1] Alfattama, S., Kanungo, P., & Bisoy, S. K. (2021). Face Recognition from Partial Face Data. APSIT 2021, IEEE. doi:10.1109/APSIT52773.2021.9641286",
    "[2] Elmahmudi, A., & Ugail, H. (2019). Deep face recognition using imperfect facial data. Future Generation Computer Systems, 99.",
    "[3] Zhang, K., et al. (2016). Joint Face Detection and Alignment Using Multitask Cascaded Convolutional Networks. IEEE SPL, 23(10).",
    "[4] Parkhi, O. M., Vedaldi, A., & Zisserman, A. (2015). Deep Face Recognition. BMVC 2015.",
    "[5] Cao, Q., et al. (2018). VGGFace2: A dataset for recognising faces across pose and age. IEEE FG 2018.",
    "[6] Huang, G. B., et al. (2007). Labeled Faces in the Wild. UMass Amherst, TR 07-49. · kaggle.com/datasets/jessicali9530/lfw-dataset",
    "[7] Thomaz, C. E., & Giraldi, G. A. (2010). FEI Face Database. Image and Vision Computing, 28(6). · kaggle.com/datasets/whizzkid/fei-face-data",
    "[8] github.com/rcmalli/keras-vggface · [9] github.com/timesler/facenet-pytorch · [10] scikit-learn, OpenCV"];
  text(s, refs.map((r, i) => ({ text: r, options: { breakLine: i < refs.length - 1 } })), { x: MX, y: 1.75, w: W - 2 * MX, h: 5.0, fontSize: 13, paraSpaceAfter: 10 });
}

{ // 33 Q&A
  const s = pres.addSlide({ masterName: "DARK", sectionTitle: currentSection });
  text(s, "Q & A", { x: MX, y: 2.0, w: 7, h: 1.5, fontSize: 72, bold: true, color: "FFFFFF" });
  text(s, "ขอบคุณครับ", { x: MX, y: 3.5, w: 7, h: 1.0, fontSize: 40, bold: true, color: "7FB0EE" });
  text(s, "Face Recognition from Partial Face · ภาคผนวก: confusion matrix รายโมเดล", { x: MX, y: 4.6, w: 7, h: 0.5, fontSize: 14, color: "8A93A6" });
  img(s, path.join(FIN, "one_person_variants.png"), 7.6, 1.6, 5.1, 4.3);
}

section("Appendix");
for (const m of METH) {
  const s = content("APPENDIX · CONFUSION MATRIX", `${m} ${MNAME[m]}:`, " 7 รูปแบบ");
  img(s, path.join(FIN, `confusion_row_${m}.png`), MX, 1.8, W - 2 * MX, 3.2);
  const vals = VARS.map((v, i) => `V${i} ${VTH[v]} ${f1(acc(m, v))}%`).join(" · ");
  card(s, MX, 5.3, W - 2 * MX, 1.4, HEX.blueSoft, HEX.blueSoft);
  text(s, [{ text: vals, options: { bold: true, breakLine: true } },
    { text: `Macro F1: ${VARS.map((v, i) => `V${i} ${f1(ACC[`${m}|${v}`].f1, 2)}`).join(" · ")} · ไฟล์ภาพขยายพร้อมชื่อคน: results/figures/final/confusion/${m}_<variant>.png`, options: { color: HEX.muted } }],
    { x: MX + 0.25, y: 5.4, w: W - 2 * MX - 0.5, h: 1.2, fontSize: 12.5, valign: "middle" });
}

(async () => {
  fs.writeFileSync(PAGES_FILE, JSON.stringify(PAGES));
  await pres.writeFile({ fileName: OUT });
  const { applyTheme } = require(process.env.SKILL + "/scripts/apply_theme.js");
  await applyTheme(OUT, THEME);
  console.log("wrote", OUT);
})();
