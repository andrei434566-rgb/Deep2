import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const source = "D:/Керн фото/!извлеченные/518431/DVD_Отчёт/Приложения/Прил.8 Описание керна_41.xlsx";
const input = await FileBlob.load(source);
const workbook = await SpreadsheetFile.importXlsx(input);
const overview = await workbook.inspect({
  kind: "workbook,sheet,table",
  maxChars: 6000,
  tableMaxRows: 12,
  tableMaxCols: 16,
  tableMaxCellChars: 150,
});
process.stdout.write(overview.ndjson);
