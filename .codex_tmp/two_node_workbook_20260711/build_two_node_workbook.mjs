import fs from "node:fs/promises";
import path from "node:path";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const repoRoot = "/Users/lucalowndes/APAC-AI-HPC-2025";
const sourceWorkbook = path.join(repoRoot, "AITaskFinalSubmission", "DeepSeek Testing.xlsx");

if (process.argv.includes("--inspect-source")) {
  const input = await FileBlob.load(sourceWorkbook);
  const workbook = await SpreadsheetFile.importXlsx(input);
  const sheets = await workbook.inspect({ kind: "sheet", include: "id,name", maxChars: 4000 });
  console.log(sheets.ndjson);
  for (let i = 0; i < workbook.worksheets.items.length; i += 1) {
    const sheet = workbook.worksheets.getItemAt(i);
    const used = sheet.getUsedRange();
    const region = await workbook.inspect({
      kind: "table",
      sheetId: sheet.name,
      range: used.address,
      maxChars: 16000,
      tableMaxRows: 100,
      tableMaxCols: 20,
      tableMaxCellChars: 100,
    });
    console.log(region.ndjson);
  }
  process.exit(0);
}

await fs.mkdir(path.join(repoRoot, "outputs", "two_node_workbook_20260711"), { recursive: true });
throw new Error("Workbook builder is not populated yet.");
