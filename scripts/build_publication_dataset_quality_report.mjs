import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const [datasetRootArg, outputDirArg] = process.argv.slice(2);
if (!datasetRootArg || !outputDirArg) {
  throw new Error("Usage: node build_publication_dataset_quality_report.mjs <dataset-root> <output-dir>");
}

const datasetRoot = path.resolve(datasetRootArg);
const outputDir = path.resolve(outputDirArg);
const manifest = JSON.parse(await fs.readFile(path.join(datasetRoot, "dataset_manifest.json"), "utf8"));
const validation = JSON.parse(await fs.readFile(path.join(datasetRoot, "validation_report.json"), "utf8"));
await fs.mkdir(outputDir, { recursive: true });

const workbook = Workbook.create();
const summary = workbook.worksheets.add("Dataset Summary");
const transformations = workbook.worksheets.add("Transformations");
const splits = workbook.worksheets.add("Split Audit");
const provenance = workbook.worksheets.add("Provenance");

const navy = "#17365D";
const blue = "#D9EAF7";
const green = "#E2F0D9";
const amber = "#FFF2CC";
const lightBorder = "#D9E2F3";

function styleTitle(sheet, range) {
  range.format = {
    fill: navy,
    font: { bold: true, color: "#FFFFFF", size: 16 },
    verticalAlignment: "center",
  };
  range.format.rowHeight = 30;
}

function styleHeader(range) {
  range.format = {
    fill: blue,
    font: { bold: true, color: "#17365D" },
    borders: { preset: "outside", style: "thin", color: lightBorder },
    verticalAlignment: "center",
    wrapText: true,
  };
  range.format.rowHeight = 28;
}

function finishSheet(sheet, usedRange, widths) {
  sheet.showGridLines = false;
  sheet.freezePanes.freezeRows(3);
  usedRange.format.font = { name: "Aptos", size: 10 };
  usedRange.format.verticalAlignment = "top";
  widths.forEach(([column, width]) => {
    sheet.getRange(`${column}:${column}`).format.columnWidth = width;
  });
}

const validationById = new Map(validation.datasets.map((item) => [item.dataset_id, item]));
const datasets = [
  ...Object.entries(manifest.real_datasets).map(([id, data]) => [id, data, "Real"]),
  ...Object.entries(manifest.synthetic_datasets).map(([id, data]) => [id, data, "Synthetic"]),
];

summary.getRange("A1:J1").merge();
summary.getRange("A1").values = [["Local fiXAIt — Publication Dataset Quality Report"]];
styleTitle(summary, summary.getRange("A1:J1"));
summary.getRange("A2:J2").merge();
summary.getRange("A2").values = [["Fixed six-dataset release; quality controls, provenance, and reproducible split assignments"]];
summary.getRange("A3:J3").values = [[
  "Dataset", "Type", "Task", "Rows", "Features", "Class distribution",
  "Missing cells", "Duplicate rows", "Validation", "Primary source",
]];
styleHeader(summary.getRange("A3:J3"));

const summaryRows = datasets.map(([id, data, kind]) => {
  const check = validationById.get(id);
  const source = data.source?.repository === "UCI Machine Learning Repository"
    ? data.source.url
    : data.source?.repository;
  return [
    data.display_name,
    kind,
    data.task,
    data.profile.rows,
    data.profile.feature_count,
    Object.entries(data.profile.class_counts).map(([key, value]) => `${key}: ${value}`).join("; "),
    data.profile.missing_cells,
    data.profile.exact_duplicate_rows,
    check.status.toUpperCase(),
    source,
  ];
});
summary.getRangeByIndexes(3, 0, summaryRows.length, 10).values = summaryRows;
summary.getRange(`A4:A${3 + summaryRows.length}`).format.wrapText = true;
summary.getRange(`F4:F${3 + summaryRows.length}`).format.wrapText = true;
summary.getRange(`J4:J${3 + summaryRows.length}`).format.wrapText = true;
summary.getRange(`A4:J${3 + summaryRows.length}`).format.rowHeight = 34;
summary.getRange(`I4:I${3 + summaryRows.length}`).format = {
  fill: green,
  font: { bold: true, color: "#375623" },
  horizontalAlignment: "center",
};
summary.getRange(`D4:H${3 + summaryRows.length}`).format.horizontalAlignment = "right";
finishSheet(summary, summary.getRange(`A1:J${3 + summaryRows.length}`), [
  ["A", 43], ["B", 14], ["C", 23], ["D", 11], ["E", 11],
  ["F", 43], ["G", 13], ["H", 14], ["I", 13], ["J", 52],
]);

transformations.getRange("A1:D1").merge();
transformations.getRange("A1").values = [["Documented Processing Decisions"]];
styleTitle(transformations, transformations.getRange("A1:D1"));
transformations.getRange("A2:D2").merge();
transformations.getRange("A2").values = [["Every change is applied before splitting and is traceable to an immutable raw snapshot."]];
transformations.getRange("A3:D3").values = [["Dataset", "Decision", "Reason / effect", "Source or scope note"]];
styleHeader(transformations.getRange("A3:D3"));
const transformationRows = [];
for (const [, data, kind] of datasets) {
  for (const step of data.processing ?? ["Regenerated at full float64 precision from the pinned source definition."]) {
    transformationRows.push([
      data.display_name,
      step,
      kind === "Real" ? "Data quality, semantic consistency, or leakage control" : "Exact provenance and numerical reproducibility",
      kind === "Real" ? data.source.doi : `XAI-Bench ${data.source.commit.slice(0, 7)}`,
    ]);
  }
  if (data.clinical_scope_note) {
    transformationRows.push([
      data.display_name,
      data.clinical_scope_note,
      "Clinical scope limitation",
      data.source.doi,
    ]);
  }
}
transformations.getRangeByIndexes(3, 0, transformationRows.length, 4).values = transformationRows;
transformations.getRange(`A4:A${3 + transformationRows.length}`).format.wrapText = true;
transformations.getRange(`B4:D${3 + transformationRows.length}`).format.wrapText = true;
finishSheet(transformations, transformations.getRange(`A1:D${3 + transformationRows.length}`), [
  ["A", 48], ["B", 80], ["C", 45], ["D", 28],
]);

splits.getRange("A1:F1").merge();
splits.getRange("A1").values = [["Reproducible Stratified Split Audit"]];
styleTitle(splits, splits.getRange("A1:F1"));
splits.getRange("A2:F2").merge();
splits.getRange("A2").values = [["All explanation methods must use the same assignment for a given seed; all preprocessing is fit on training rows only."]];
splits.getRange("A3:F3").values = [["Dataset", "Seed", "Train rows", "Test rows", "Test class distribution", "Split file SHA-256"]];
styleHeader(splits.getRange("A3:F3"));
const splitRows = [];
for (const [id, splitData] of Object.entries(manifest.split_assignments)) {
  const displayName = datasets.find(([datasetId]) => datasetId === id)[1].display_name;
  for (const [seed, counts] of Object.entries(splitData.seeds)) {
    splitRows.push([
      displayName,
      Number(seed),
      counts.train_rows,
      counts.test_rows,
      Object.entries(counts.test_class_counts).map(([key, value]) => `${key}: ${value}`).join("; "),
      splitData.sha256,
    ]);
  }
}
splits.getRangeByIndexes(3, 0, splitRows.length, 6).values = splitRows;
splits.getRange(`A4:A${3 + splitRows.length}`).format.wrapText = true;
splits.getRange(`F4:F${3 + splitRows.length}`).format = { font: { name: "Consolas", size: 8 }, wrapText: true };
finishSheet(splits, splits.getRange(`A1:F${3 + splitRows.length}`), [
  ["A", 48], ["B", 10], ["C", 13], ["D", 12], ["E", 32], ["F", 68],
]);

provenance.getRange("A1:D1").merge();
provenance.getRange("A1").values = [["Sources, Licenses, and Reproducibility"]];
styleTitle(provenance, provenance.getRange("A1:D1"));
provenance.getRange("A2:D2").merge();
provenance.getRange("A2").values = [["Retain README.md, CITATION.bib, dataset_manifest.json, and LICENSE_XAI_BENCH.txt with the release."]];
provenance.getRange("A3:D3").values = [["Dataset", "Source", "License", "Pinned identifier / DOI"]];
styleHeader(provenance.getRange("A3:D3"));
const provenanceRows = datasets.map(([, data]) => [
  data.display_name,
  data.source.repository,
  data.source.license,
  data.source.doi ?? data.source.commit,
]);
provenance.getRangeByIndexes(3, 0, provenanceRows.length, 4).values = provenanceRows;
const noteRow = 5 + provenanceRows.length;
provenance.getRange(`A${noteRow}:D${noteRow}`).merge();
provenance.getRange(`A${noteRow}`).values = [[manifest.publication_scope_note]];
provenance.getRange(`A${noteRow}:D${noteRow}`).format = { fill: amber, wrapText: true, font: { italic: true, color: "#7F6000" } };
provenance.getRange(`A${noteRow}:D${noteRow}`).format.rowHeight = 44;
finishSheet(provenance, provenance.getRange(`A1:D${noteRow}`), [
  ["A", 48], ["B", 55], ["C", 18], ["D", 55],
]);

const reportPath = path.join(outputDir, "publication_dataset_quality_report.xlsx");
const exported = await SpreadsheetFile.exportXlsx(workbook);
await exported.save(reportPath);

for (const sheetName of ["Dataset Summary", "Transformations", "Split Audit", "Provenance"]) {
  const preview = await workbook.render({
    sheetName,
    autoCrop: "all",
    scale: 1,
    format: "png",
  });
  const previewName = sheetName.toLowerCase().replaceAll(" ", "_");
  await fs.writeFile(
    path.join(outputDir, `publication_dataset_quality_report_${previewName}.png`),
    new Uint8Array(await preview.arrayBuffer()),
  );
}

const inspection = await workbook.inspect({
  kind: "workbook,sheet,region",
  maxChars: 6000,
  tableMaxRows: 10,
  tableMaxCols: 10,
  tableMaxCellChars: 100,
});
console.log(inspection.ndjson);
const formulaErrors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 100 },
  summary: "final formula error scan",
});
console.log(formulaErrors.ndjson);
console.log(JSON.stringify({ reportPath, sheetCount: 4 }, null, 2));
