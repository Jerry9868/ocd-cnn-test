// Builds reports/OCD_ML_report.pptx. Kept for reference: it needs Node, the
// pptxgenjs npm package (4.x) and the apply_theme.js helper from the machine it
// was written on (path below), so it will not run on a machine without them.
// There, edit the .pptx directly instead.
//   usage: NODE_PATH=<folder containing pptxgenjs> node build_report.js OCD_ML_report.pptx
const pptxgen = require("pptxgenjs");
const { applyTheme } = require(
  "C:/Users/Jerry/.claude/skills/synced/f4694ae0-5564-4c11-af98-e7ffd47f9530_a8f3557f-76a7-4192-b7f7-242139ea7efb/pptx/scripts/apply_theme.js"
);

const OUT = process.argv[2];
const FONT = "Calibri";
const THEME = {
  name: "OCD ML Report",
  headFontFace: FONT,
  bodyFontFace: FONT,
  colors: {
    dk1: "13262B", lt1: "FFFFFF", dk2: "0B3C49", lt2: "EAF2F2",
    accent1: "0E7C7B", accent2: "F2A541", accent3: "5B6B70",
    accent4: "C8553D", accent5: "8FB8B6", accent6: "D3E0E0",
    hlink: "0E7C7B", folHlink: "5B6B70",
  },
};
const H = THEME.colors;

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9";
pres.theme = { headFontFace: FONT, bodyFontFace: FONT };
pres.title = "Mueller Matrix Spectra and Machine Learning for CD";
pres.author = "Jerry";
const C = pres.SchemeColor;
const FOOTER = "Synthetic-data results · 2026-10-10";

pres.defineSlideMaster({
  title: "TITLE_DARK",
  background: { color: H.dk2 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.5, y: 1.45, w: 5.6, h: 1.6, fontSize: 30, bold: true, color: C.background1, valign: "top", align: "left", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.5, y: 3.2, w: 5.6, h: 0.9, fontSize: 18, color: C.accent5, valign: "top", align: "left", margin: 0 }, text: "" } },
  ],
});
for (const [name, bg, titleColor, footColor] of [
  ["CONTENT", H.lt1, C.text1, C.accent3],
  ["CONTENT_DARK", H.dk2, C.background1, C.accent5],
]) {
  pres.defineSlideMaster({
    title: name,
    background: { color: bg },
    objects: [
      { placeholder: { options: { name: "title", type: "title", x: 0.5, y: 0.32, w: 9.0, h: 0.75, fontSize: 28, bold: true, color: titleColor, valign: "middle", align: "left", margin: 0 }, text: "" } },
      { text: { text: FOOTER, options: { x: 0.5, y: 5.2, w: 5, h: 0.25, fontSize: 10, color: footColor, margin: 0 } } },
    ],
    slideNumber: { x: 9.0, y: 5.2, w: 0.5, h: 0.25, fontSize: 10, color: footColor, align: "right" },
  });
}

// ---- helpers ---------------------------------------------------------------
let seq = 0;
const nm = (base) => `${base} ${++seq}`;

function text(slide, t, o) {
  slide.addText(t, { isTextBox: true, margin: 0, valign: "top", color: C.text1, fontSize: 14, objectName: nm("Text"), ...o });
}
function card(slide, x, y, w, h, fill = C.background2) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x, y, w, h, rectRadius: 0.08, fill: { color: fill }, line: { color: fill, width: 0 }, objectName: nm("Card"),
  });
}
function badge(slide, x, y, label, fill = C.accent1, color = C.background1) {
  slide.addShape(pres.shapes.OVAL, { x, y, w: 0.34, h: 0.34, fill: { color: fill }, line: { color: fill, width: 0 }, objectName: nm("Badge") });
  slide.addText(String(label), { isTextBox: true, x, y, w: 0.34, h: 0.34, margin: 0, align: "center", valign: "middle", fontSize: 12, bold: true, color, objectName: nm("Badge label") });
}
function caption(slide, t, x, y, w) {
  text(slide, t, { x, y, w, h: 0.3, fontSize: 10, color: C.accent3 });
}
const axis = {
  catAxisLabelColor: H.accent3, valAxisLabelColor: H.accent3,
  catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt",
  catAxisLabelFontSize: 11, valAxisLabelFontSize: 10,
  dataLabelFontFace: "+mn-lt", dataLabelFontSize: 10, dataLabelColor: H.dk1,
  legendFontFace: "+mn-lt", legendFontSize: 11, legendColor: H.accent3,
  titleFontFace: "+mn-lt", titleFontSize: 12, titleColor: H.accent3,
  valGridLine: { color: H.accent6, size: 0.5 }, catGridLine: { style: "none" },
  valAxisLineShow: false, valAxisMinVal: 0,
  showValue: true, dataLabelFormatCode: "0.000",
  showTitle: true,
};
// Numbered commentary column: [{ head, body }]
function points(slide, items, x, y, w, gap = 1.05) {
  items.forEach((it, i) => {
    badge(slide, x, y + i * gap + 0.02, i + 1);
    text(slide, [
      { text: it.head, options: { bold: true, fontSize: 14, breakLine: true } },
      { text: it.body, options: { fontSize: 12, color: C.accent3 } },
    ], { x: x + 0.48, y: y + i * gap, w: w - 0.48, h: gap - 0.1, paraSpaceAfter: 2 });
  });
}

// ---- 1. title ----------------------------------------------------------------
pres.addSection({ title: "Summary" });
{
  const s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: "Summary" });
  s.addText([{ text: "Mueller Matrix Spectra", options: { breakLine: true } }, { text: "to CD by Machine Learning" }], { placeholder: "title" });
  s.addText("Results on synthetic data, and what is missing before real data", { placeholder: "body" });
  text(s, "2026-10-10 · Every number is from synthetic spectra; no measured data has been used yet", { x: 0.5, y: 4.75, w: 6.2, h: 0.3, fontSize: 12, color: C.accent5 });
  // Motif: the 4 x 4 Mueller matrix plus the two normal-incidence spectra.
  const labels = [];
  for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) labels.push(`M${i}${j}`);
  const shade = [0, 55, 70, 40, 60, 20, 45, 75, 35, 65, 10, 50, 70, 30, 55, 25];
  const x0 = 6.55, y0 = 1.0, cell = 0.64, pitch = 0.74;
  labels.forEach((lab, k) => {
    const x = x0 + (k % 4) * pitch, y = y0 + Math.floor(k / 4) * pitch;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w: cell, h: cell, rectRadius: 0.06, fill: { color: C.accent1, transparency: shade[k] }, line: { color: C.accent1, width: 0 }, objectName: nm("Mueller cell") });
    s.addText(lab, { isTextBox: true, x, y, w: cell, h: cell, margin: 0, align: "center", valign: "middle", fontSize: 10, color: C.background1, objectName: nm("Mueller label") });
  });
  ["TE", "TM"].forEach((lab, k) => {
    const x = x0 + k * (2 * pitch), y = y0 + 4 * pitch + 0.1, w = 2 * pitch - (pitch - cell);
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h: 0.42, rectRadius: 0.06, fill: { color: C.accent2 }, line: { color: C.accent2, width: 0 }, objectName: nm("Reference cell") });
    s.addText(lab, { isTextBox: true, x, y, w, h: 0.42, margin: 0, align: "center", valign: "middle", fontSize: 11, bold: true, color: C.text2, objectName: nm("Reference label") });
  });
  s.addNotes("Cover. The grid shows the 18 input channels: 16 Mueller elements measured at 65 degrees oblique incidence, plus TE and TM at normal incidence.");
}

// ---- 2. summary ----------------------------------------------------------------
{
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Summary" });
  s.addText("At the theoretical limit, on synthetic data only", { placeholder: "title" });
  const stats = [
    ["< 0.01%", "Errors above 0.3 nm", "0.65% with the CNN alone", C.accent1],
    ["0.067 nm", "CD RMSE after refinement", "Equal to the theoretical lower bound of 0.067 nm", C.accent1],
    ["0", "Real samples verified", "Every conclusion still needs real data", C.accent4],
  ];
  stats.forEach(([big, head, sub, color], i) => {
    const x = 0.5 + i * 3.1;
    card(s, x, 1.3, 2.8, 2.35);
    text(s, big, { x: x + 0.2, y: 1.45, w: 2.4, h: 0.9, fontSize: 32, bold: true, color, valign: "middle" });
    text(s, head, { x: x + 0.2, y: 2.4, w: 2.4, h: 0.5, fontSize: 14, bold: true });
    text(s, sub, { x: x + 0.2, y: 2.95, w: 2.4, h: 0.6, fontSize: 12, color: C.accent3 });
  });
  text(s, [
    { text: "Helps: ", options: { bold: true, color: C.accent1 } },
    { text: "refinement (largest gain), MLP in place of CNN, cross-channel information", options: { breakLine: true } },
    { text: "Little effect: ", options: { bold: true, color: C.accent3 } },
    { text: "deeper or wider networks, ensembles, more than 5,600 training spectra", options: { breakLine: true } },
    { text: "Largest gap: ", options: { bold: true, color: C.accent4 } },
    { text: "real file format unconfirmed, and far more labels needed than TEM usually provides" },
  ], { x: 0.5, y: 3.9, w: 9.0, h: 1.1, fontSize: 14, paraSpaceAfter: 4 });
  s.addNotes("Three figures: the best synthetic result, the share of samples beyond 0.3 nm, and the absence of any real-data verification. Acceptance: every sample within 0.3 nm first, RMSE second.");
}

// ---- 3. method ----------------------------------------------------------------
pres.addSection({ title: "Method and results" });
{
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Method and results" });
  s.addText("Method: network estimate, then a model fit", { placeholder: "title" });
  const steps = [
    ["Spectrum", "18 channels × 101 wavelengths", C.background2, C.text1],
    ["Network", "MLP or multi-channel 1D CNN", C.background2, C.text1],
    ["Initial CD", "One forward pass", C.background2, C.text1],
    ["Refinement", "Least-squares fit to the forward model", C.accent1, C.background1],
    ["CD ± error", "Plus a misfit warning", C.accent1, C.background1],
  ];
  steps.forEach(([head, sub, fill, color], i) => {
    const x = 0.5 + i * 1.85;
    card(s, x, 1.3, 1.6, 1.6, fill);
    text(s, head, { x: x + 0.1, y: 1.5, w: 1.4, h: 0.4, fontSize: 15, bold: true, color, align: "center" });
    text(s, sub, { x: x + 0.1, y: 2.0, w: 1.4, h: 0.8, fontSize: 12, color, align: "center" });
    if (i < steps.length - 1) {
      s.addShape(pres.shapes.CHEVRON, { x: x + 1.65, y: 1.99, w: 0.15, h: 0.22, fill: { color: C.accent5 }, line: { color: C.accent5, width: 0 }, objectName: nm("Arrow") });
    }
  });
  const notes = [
    ["No RCWA needed", "A second network is trained on the same data in reverse, CD to spectrum. The regression averages the noise out."],
    ["Yardstick: Cramér–Rao bound", "The smallest error any unbiased estimator can reach at a given noise level. It shows how much room is left."],
    ["Data: synthetic spectra", "CD 20–80 nm, noise 0.01. Not a physical simulation; used to validate the pipeline and compare methods."],
  ];
  notes.forEach(([head, body], i) => {
    const x = 0.5 + i * 3.1;
    text(s, head, { x, y: 3.25, w: 2.8, h: 0.35, fontSize: 15, bold: true, color: C.accent1 });
    text(s, body, { x, y: 3.7, w: 2.8, h: 1.3, fontSize: 14, color: C.accent3 });
  });
  s.addNotes("Two stages: the network gives an initial CD, then Gauss-Newton fits it against a forward model learned from the same training data.");
}

// ---- 4. architecture comparison ---------------------------------------------
{
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Method and results" });
  s.addText("MLP plus refinement reaches the bound", { placeholder: "title" });
  const labels = ["CNN baseline", "CNN, 3-model ensemble", "CNN, noise augmentation", "MLP", "MLP, 3-model ensemble", "MLP + refinement", "Theoretical bound"];
  const values = [0.105, 0.097, 0.095, 0.077, 0.074, 0.068, 0.067];
  s.addChart(pres.charts.BAR, [{ name: "CD RMSE (nm)", labels, values }], {
    ...axis, x: 0.4, y: 1.2, w: 5.6, h: 3.7, barDir: "bar", barGapWidthPct: 45,
    chartColors: [H.accent5, H.accent5, H.accent5, H.accent1, H.accent1, H.accent1, H.accent2],
    catAxisOrientation: "maxMin", valAxisHidden: true, valGridLine: { style: "none" },
    dataLabelPosition: "outEnd", showLegend: false, title: "CD RMSE (nm), lower is better",
    objectName: nm("Chart"),
  });
  points(s, [
    { head: "The CNN is noise-sensitive", body: "1.57 times the bound. Normalization, pooling, width and learning rate all land between 0.102 and 0.132 nm." },
    { head: "The MLP gets to 1.15 times", body: "Possibly because synthetic spectra lack local structure; this must be re-tested on real spectra." },
    { head: "Refinement gets to 1.02 times", body: "The ranking is the same for three other synthetic physics settings." },
  ], 6.25, 1.35, 3.25, 1.2);
  caption(s, "Only CD varies; 5,600 training spectra; scored on 20,000 fresh spectra", 0.5, 4.9, 5.5);
  s.addNotes("Main comparison. Bound 0.067 nm; CNN 0.105, MLP 0.077, with refinement 0.068.");
}

// ---- 5. sample size ------------------------------------------------------------
{
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Method and results" });
  s.addText("Error rises sharply below about 700 training samples", { placeholder: "title" });
  const labels = ["106", "210", "420", "700", "5,600"];
  s.addChart(pres.charts.BAR, [
    { name: "MLP", labels, values: [0.305, 0.240, 0.178, 0.132, 0.077] },
    { name: "MLP + refinement", labels, values: [0.502, 0.413, 0.161, 0.089, 0.068] },
  ], {
    ...axis, x: 0.4, y: 1.2, w: 5.6, h: 3.7, barDir: "col", barGapWidthPct: 45,
    chartColors: [H.accent5, H.accent1], dataLabelPosition: "outEnd", dataLabelFormatCode: "0.00",
    showLegend: true, legendPos: "b", title: "CD RMSE (nm) by number of training samples",
    objectName: nm("Chart"),
  });
  points(s, [
    { head: "Refinement hurts below 700", body: "The forward model is not accurate enough yet. Training compares both on validation data and picks automatically." },
    { head: "The CNN needs more data", body: "0.290 nm at 700 samples, 0.105 nm at 5,600, and flat beyond that." },
    { head: "What it means for real data", body: "With only tens of TEM values, expect errors above 0.3 nm; another source of labels would be needed." },
  ], 6.25, 1.35, 3.25, 1.2);
  caption(s, "Only CD varies; labels without error; theoretical bound 0.067 nm", 0.5, 4.9, 5.5);
  s.addNotes("Sample count is the key constraint for real data. The third point is inferred from the synthetic results and not yet confirmed on real data.");
}

// ---- 6. wavelength offset / error bar -----------------------------------------
{
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Method and results" });
  s.addText("Refinement adds error bars and absorbs offsets", { placeholder: "title" });
  const labels = ["0 nm", "0.2 nm", "0.5 nm", "1.0 nm"];
  s.addChart(pres.charts.BAR, [
    { name: "MLP", labels, values: [0.077, 0.099, 0.176, 0.329] },
    { name: "Refinement, offset fitted", labels, values: [0.072, 0.072, 0.071, 0.071] },
  ], {
    ...axis, x: 0.4, y: 1.2, w: 5.6, h: 3.7, barDir: "col", barGapWidthPct: 60,
    chartColors: [H.accent5, H.accent1], dataLabelPosition: "outEnd",
    showLegend: true, legendPos: "b", title: "CD RMSE (nm) by wavelength-axis offset",
    objectName: nm("Chart"),
  });
  points(s, [
    { head: "The offset is fitted too", body: "A 1 nm wavelength offset quadruples the network's error; fitting the offset alongside CD removes the effect." },
    { head: "The error bar is calibrated", body: "Actual error divided by predicted standard deviation has a spread of 1.03, against an ideal 1.00." },
    { head: "Unusual spectra are flagged", body: "With doubled noise 100% of spectra are flagged, against 0.02% of normal ones." },
  ], 6.25, 1.35, 3.25, 1.2);
  caption(s, "Only CD varies; 5,600 training spectra", 0.5, 4.9, 5.5);
  s.addNotes("Beyond accuracy, refinement gives a per-spectrum error bar, a misfit warning, and robustness to wavelength calibration offsets.");
}

// ---- 7. label noise -------------------------------------------------------------
{
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Method and results" });
  s.addText("TEM label error of 0.3 nm is tolerable; 1 nm is not", { placeholder: "title" });
  const cases = [
    ["No label error", "0.077", "0.068", "Close to the 0.067 nm bound"],
    ["Label error 0.3 nm", "0.117", "0.076", "Refinement is barely affected"],
    ["Label error 1.0 nm", "0.258", "0.178", "Both clearly worse"],
  ];
  cases.forEach(([head, net, ref, note], i) => {
    const x = 0.5 + i * 3.1;
    card(s, x, 1.3, 2.8, 2.15);
    text(s, head, { x: x + 0.2, y: 1.42, w: 2.4, h: 0.35, fontSize: 14, bold: true });
    text(s, net, { x: x + 0.2, y: 1.85, w: 1.2, h: 0.6, fontSize: 28, bold: true, color: C.accent3, valign: "middle" });
    text(s, ref, { x: x + 1.4, y: 1.85, w: 1.2, h: 0.6, fontSize: 28, bold: true, color: C.accent1, valign: "middle" });
    text(s, "MLP", { x: x + 0.2, y: 2.45, w: 1.2, h: 0.3, fontSize: 11, color: C.accent3 });
    text(s, "With refinement", { x: x + 1.4, y: 2.45, w: 1.2, h: 0.3, fontSize: 11, color: C.accent1 });
    text(s, note, { x: x + 0.2, y: 2.9, w: 2.4, h: 0.4, fontSize: 12, color: C.accent3 });
  });
  const why = [
    ["Label error passes to the network", "The network learns the erroneous labels; the error is not averaged out."],
    ["Refinement averages it out", "With 16,800 training spectra the 0.3 nm case reaches 0.072 nm."],
    ["Range edges biased at 1 nm", "The interior stays at 0.077 nm, so train on a wider range than is used."],
  ];
  why.forEach(([head, body], i) => {
    const x = 0.5 + i * 3.1;
    text(s, head, { x, y: 3.7, w: 2.8, h: 0.3, fontSize: 14, bold: true, color: C.accent1 });
    text(s, body, { x, y: 4.05, w: 2.8, h: 0.6, fontSize: 12, color: C.accent3 });
  });
  caption(s, "CD RMSE (nm). Only CD varies; 5,600 training spectra; test labels without error", 0.5, 4.85, 9.0);
  s.addNotes("TEM label error is simulated as Gaussian. The size and distribution of real TEM error still need to be established.");
}

// ---- 8. nuisance parameters ---------------------------------------------------
{
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Method and results" });
  s.addText("Labels for other parameters are valuable", { placeholder: "title" });
  const labels = ["CNN", "MLP", "MLP, multi-task", "Refinement, CD labels only", "Refinement, all labels", "Theoretical bound"];
  const values = [0.154, 0.133, 0.122, 0.125, 0.096, 0.090];
  s.addChart(pres.charts.BAR, [{ name: "CD RMSE (nm)", labels, values }], {
    ...axis, x: 0.4, y: 1.2, w: 5.6, h: 3.7, barDir: "bar", barGapWidthPct: 45,
    chartColors: [H.accent5, H.accent5, H.accent5, H.accent1, H.accent1, H.accent2],
    catAxisOrientation: "maxMin", valAxisHidden: true, valGridLine: { style: "none" },
    dataLabelPosition: "outEnd", showLegend: false, title: "CD RMSE (nm) with 3 nuisance parameters",
    objectName: nm("Chart"),
  });
  points(s, [
    { head: "The bound itself rises", body: "From 0.067 to 0.090 nm. Three extra parameters stand in for height, sidewall angle and film thickness." },
    { head: "With their labels: near the bound", body: "Refinement reaches 0.096 nm; the multi-task network 0.122 nm." },
    { head: "CD labels only: position matters", body: "0.098 nm in the central 70% of the training range, but 0.187 nm in the outer 3%, worse than the network's 0.161." },
  ], 6.25, 1.35, 3.25, 1.2);
  caption(s, "16,800–33,600 training spectra; the nuisance parameters are generic stand-ins, not physical models", 0.5, 4.9, 6.5);
  s.addNotes("The nuisance-parameter case is closest to real measurement. With CD labels only, the manifold method helps only when the training range is clearly wider than the range in use.");
}

// ---- 9. what helps ------------------------------------------------------------
pres.addSection({ title: "Conclusions and gaps" });
{
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Conclusions and gaps" });
  s.addText("Which ML methods help, on synthetic data", { placeholder: "title" });
  const rows = [
    ["Refinement (learned forward model)", "1.57× bound → 1.02×", "Helps most", H.accent1],
    ["MLP in place of CNN", "1.57× bound → 1.15×", "Helps; re-test on real data", H.accent1],
    ["Cross-channel vs. per-channel convolution", "0.105 vs. 0.138 nm", "Cross-channel helps", H.accent1],
    ["Noise augmentation, ensembles", "0.105 → 0.095 / 0.097 nm", "Small gain", H.accent3],
    ["Multi-task: also predict other parameters", "0.133 → 0.122 nm", "Small gain; needs labels", H.accent3],
    ["Depth, width, normalization, pooling, more data", "No clear change", "No effect", H.accent4],
    ["Channel attention, 2D / 3D CNN", "0.105 → 0.108 / 0.111 / 0.119 nm", "No effect", H.accent4],
    ["First-derivative features", "CNN 0.105 → 0.121, MLP 0.077 → 0.096", "Worse", H.accent4],
  ];
  const head = (t) => ({ text: t, options: { bold: true, color: H.lt1, fill: { color: H.dk2 }, fontSize: 12 } });
  const table = [[head("Method"), head("Change in CD RMSE"), head("Verdict")]];
  rows.forEach(([a, b, c, color], i) => {
    const fill = { color: i % 2 ? H.lt1 : H.lt2 };
    table.push([
      { text: a, options: { fill, color: H.dk1, fontSize: 12 } },
      { text: b, options: { fill, color: H.dk1, fontSize: 12 } },
      { text: c, options: { fill, color, bold: true, fontSize: 12 } },
    ]);
  });
  s.addTable(table, {
    x: 0.5, y: 1.25, w: 9.0, colW: [3.6, 3.2, 2.2], rowH: 0.39, valign: "middle",
    border: { type: "none" }, fontFace: FONT, objectName: nm("Table"),
  });
  caption(s, "Original question (is cross-channel information among the Mueller elements useful?): yes within a CNN, but an MLP reading every channel directly does better", 0.5, 4.85, 9.0);
  s.addNotes("The attention gates all settled near 1, so every synthetic channel is equally informative. The 2D and 3D CNNs use narrower layers because of CPU cost.");
}

// ---- 10. gaps ------------------------------------------------------------------
{
  const s = pres.addSlide({ masterName: "CONTENT_DARK", sectionTitle: "Conclusions and gaps" });
  s.addText("Real data: code is ready, data is missing", { placeholder: "title" });
  const cols = [
    ["Data still needed", [
      ["One real spectrum export", "Confirms format, wavelength range, channel count"],
      ["Number of spectrum–TEM pairs", "Below a few hundred, another label source is needed"],
      ["Repeat measurements of one site", "To estimate the real measurement noise"],
      ["What comes with each TEM value", "Wafer ID, plus height, sidewall angle and similar"],
      ["A wide enough sampling range", "Train on a wider range than will be used"],
    ]],
    ["Code ready, awaiting real data", [
      ["Training on real data", "A folder of spectra plus a reference-CD table"],
      ["Cross-validation", "Grouped by wafer; suited to small sample counts"],
      ["Method comparison tool", "Every method scored on the same splits"],
      ["Install and handoff documents", "Pinned versions, intranet install, notes for the next agent"],
      ["Still to do", "Re-run every comparison on real data"],
    ]],
  ];
  cols.forEach(([head, items], c) => {
    const x = 0.5 + c * 4.65;
    text(s, head, { x, y: 1.25, w: 4.35, h: 0.4, fontSize: 16, bold: true, color: C.accent2 });
    items.forEach(([what, why], i) => {
      const y = 1.8 + i * 0.66;
      badge(s, x, y + 0.02, i + 1, C.accent1, C.background1);
      text(s, what, { x: x + 0.48, y, w: 3.87, h: 0.3, fontSize: 14, bold: true, color: C.background1 });
      text(s, why, { x: x + 0.48, y: y + 0.3, w: 3.87, h: 0.28, fontSize: 12, color: C.accent5 });
    });
  });
  s.addNotes("Acceptance is set: every sample within 0.3 nm first, RMSE second. The real-data pipeline has so far only been tested on files written from synthetic spectra.");
}

(async () => {
  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("wrote", OUT);
})();
