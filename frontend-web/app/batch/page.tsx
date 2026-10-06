"use client";

import { CircleAlert, Download, FileCheck2, FolderOpen, ListOrdered, ShieldCheck, Upload } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import {
  ApiError, BatchPredictionResponse, batchTemplateUrl, fetchModelInfo, predictBatch,
} from "@/lib/api";
import { COLUMN_GROUPS, PRIORITY_BADGE, explainFieldError } from "@/lib/constants";

type RowErrorDetail = {
  message: string;
  errors?: { row: number; field: string; message: string; value?: string }[];
  total_errors?: number;
  fileName: string;
};
type Priority = "All" | "High" | "Medium" | "Low";

const MISSING_COLUMNS_PREFIX = "Missing required column(s): ";

function joinWithAnd(items: string[]): string {
  if (items.length === 1) return items[0];
  if (items.length === 2) return `${items[0]} and ${items[1]}`;
  return `${items.slice(0, -1).join(", ")}, and ${items[items.length - 1]}`;
}

function StatusPill({ bg, text, children }: { bg: string; text: string; children: React.ReactNode }) {
  return (
    <span className="flex gap-[6px] items-center px-[10px] py-[4px] rounded-full w-fit" style={{ backgroundColor: bg }}>
      <span className="size-[5px] rounded-full" style={{ backgroundColor: text }} />
      <span className="font-semibold text-[13px]" style={{ color: text }}>{children}</span>
    </span>
  );
}

export default function BatchUploadPage() {
  const [fileName, setFileName] = useState<string | null>(null);
  const [response, setResponse] = useState<BatchPredictionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [rowErrors, setRowErrors] = useState<RowErrorDetail | null>(null);
  const [fileTypeError, setFileTypeError] = useState<string | null>(null);
  const [missingColumns, setMissingColumns] = useState<{ fileName: string; columns: string[] } | null>(null);
  const [loading, setLoading] = useState(false);
  const [filter, setFilter] = useState<Priority>("All");
  const [threshold, setThreshold] = useState<number | null>(null);
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    fetchModelInfo().then((i) => setThreshold(i.threshold)).catch(() => {});
  }, []);

  async function handleFile(file: File) {
    if (!file.name.toLowerCase().endsWith(".csv")) {
      setFileTypeError(file.name);
      setError(null);
      setRowErrors(null);
      setMissingColumns(null);
      setResponse(null);
      return;
    }
    setFileTypeError(null);
    setLoading(true);
    setError(null);
    setRowErrors(null);
    setMissingColumns(null);
    setResponse(null);
    try {
      const result = await predictBatch(file);
      setResponse(result);
      setFileName(file.name);
    } catch (err) {
      if (err instanceof ApiError && typeof err.detail === "object" && err.detail !== null) {
        setRowErrors({ ...(err.detail as RowErrorDetail), fileName: file.name });
      } else if (err instanceof ApiError && typeof err.detail === "string" && err.detail.startsWith(MISSING_COLUMNS_PREFIX)) {
        setMissingColumns({ fileName: file.name, columns: err.detail.slice(MISSING_COLUMNS_PREFIX.length).split(", ") });
      } else if (err instanceof ApiError) {
        setError(String(err.detail));
      } else {
        setError("Could not reach the API.");
      }
    } finally {
      setLoading(false);
    }
  }

  function onFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (file) handleFile(file);
  }

  function onDrop(e: React.DragEvent<HTMLDivElement>) {
    e.preventDefault();
    setDragging(false);
    const file = e.dataTransfer.files?.[0];
    if (file) handleFile(file);
  }

  function onReset() {
    setResponse(null);
    setFileName(null);
    setError(null);
    setRowErrors(null);
    setFileTypeError(null);
    setMissingColumns(null);
    setFilter("All");
    if (inputRef.current) inputRef.current.value = "";
    inputRef.current?.click();
  }

  const shown = response
    ? filter === "All" ? response.results : response.results.filter((r) => r.priority === filter)
    : [];

  const counts = response
    ? { High: response.results.filter((r) => r.priority === "High").length,
        Medium: response.results.filter((r) => r.priority === "Medium").length,
        Low: response.results.filter((r) => r.priority === "Low").length,
        Call: response.results.filter((r) => r.recommendation === "Call").length }
    : { High: 0, Medium: 0, Low: 0, Call: 0 };

  function downloadRankedCsv() {
    if (!response) return;
    const extraKeys = Object.keys(shown[0] ?? {}).filter(
      (k) => !["rank", "probability", "predicted_class", "recommendation", "priority"].includes(k)
    );
    const header = [...extraKeys, "rank", "probability", "predicted_class", "recommendation", "priority"];
    const lines = [header.join(",")];
    for (const row of shown) lines.push(header.map((key) => String(row[key] ?? "")).join(","));
    const blob = new Blob([lines.join("\n")], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "ranked_call_list.csv";
    a.click();
    URL.revokeObjectURL(url);
  }

  function downloadErrorReport() {
    if (!rowErrors?.errors) return;
    const lines = ["csv_row,column,value,how_to_fix"];
    for (const e of rowErrors.errors) {
      lines.push([e.row, e.field, e.value ?? "", explainFieldError(e.field)]
        .map((v) => `"${String(v).replace(/"/g, '""')}"`).join(","));
    }
    const blob = new Blob([lines.join("\n")], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "error_report.csv";
    a.click();
    URL.revokeObjectURL(url);
  }

  const high = threshold != null ? Math.round(threshold * 100) : null;
  const medium = threshold != null ? Math.round((threshold / 2) * 100) : null;

  return (
    <div className="flex flex-col items-start w-full">
      <div className="flex flex-col gap-[28px] items-start p-[40px] w-full max-w-[1200px] mx-auto">
        <div className="flex flex-col gap-[8px] items-start w-full">
          <p className="font-semibold text-[#65736d] text-[13px] uppercase">Batch upload</p>
          <p className="font-semibold text-[#23362f] text-[34px] leading-[1.4]">Build your call list</p>
          <p className="text-[#65736d] text-[16px] leading-[1.4]">
            Upload customer details, then review a ranked list to focus your outreach.
          </p>
        </div>

        <input ref={inputRef} type="file" accept=".csv" onChange={onFileChange} className="hidden" />

        <div className="bg-white border border-[#dde4df] rounded-[10px] p-[24px] flex flex-col gap-[18px] w-full">
          <div className="flex items-center justify-between w-full">
            <p className="flex-1 font-semibold text-[#23362f] text-[21px]">
              {fileName ? "Customer list uploaded" : "Upload a customer list"}
            </p>
            <a href={batchTemplateUrl()} download
              className="bg-white border border-[#dde4df] flex gap-[8px] items-center px-[18px] py-[10px] rounded-[6px]">
              <Download size={20} className="text-[#176547]" />
              <span className="font-semibold text-[16px] text-[#176547]">Download CSV template</span>
            </a>
          </div>

          {fileName ? (
            <div className="bg-[#eaf3ee] flex gap-[14px] items-center p-[16px] rounded-[6px] w-full">
              <FileCheck2 size={24} className="text-[#176547] shrink-0" />
              <div className="flex flex-1 flex-col gap-[2px] min-w-0">
                <p className="font-semibold text-[#23362f] text-[16px]">{fileName}</p>
                <p className="text-[#65736d] text-[14px]">
                  {response?.count ?? 0} customer rows · all 19 required columns
                </p>
              </div>
              <StatusPill bg="#eaf3ee" text="#176547">Processed</StatusPill>
              <button onClick={onReset}
                className="bg-white border border-[#dde4df] px-[18px] py-[10px] rounded-[6px] font-semibold text-[16px] text-[#176547]">
                Upload another list
              </button>
            </div>
          ) : (
            <div
              onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
              onDragLeave={() => setDragging(false)}
              onDrop={onDrop}
              className={`flex gap-[24px] items-center p-[28px] rounded-[6px] w-full border-[1.5px] border-dashed ${
                dragging ? "bg-[#eaf3ee] border-[#176547]" : "bg-[#f4f6f5] border-[#dde4df]"
              }`}>
              <div className="bg-white flex items-center justify-center rounded-[10px] shrink-0 size-[56px]">
                <Upload size={26} className="text-[#176547]" />
              </div>
              <div className="flex flex-1 flex-col gap-[4px] min-w-0">
                <p className="font-semibold text-[#23362f] text-[20px]">
                  {loading ? "Scoring every client…" : "Drag and drop your CSV here"}
                </p>
                <p className="text-[#65736d] text-[15px]">One row = one customer. Use the template to get started.</p>
                <p className="text-[#65736d] text-[13px]">CSV files only · include the header row</p>
              </div>
              <button onClick={() => inputRef.current?.click()} disabled={loading}
                className="bg-[#176547] border border-[#176547] flex gap-[8px] items-center px-[18px] py-[10px] rounded-[6px] shrink-0 disabled:opacity-60">
                <FolderOpen size={20} className="text-white" />
                <span className="font-semibold text-[16px] text-white">Browse files</span>
              </button>
            </div>
          )}
        </div>

        {fileTypeError && (
          <div className="bg-[#faedec] flex gap-[12px] items-start p-[16px] rounded-[6px] w-full">
            <CircleAlert size={20} className="text-[#9a4948] shrink-0" />
            <div className="flex flex-col gap-[3px]">
              <p className="font-semibold text-[#9a4948] text-[16px]">Wrong file type — CSV required</p>
              <p className="text-[#9a4948] text-[14px] leading-[1.4]">
                {fileTypeError} is not supported. Save the spreadsheet as a CSV and upload it again.
                No predictions were generated.
              </p>
            </div>
          </div>
        )}

        {missingColumns && (
          <div className="bg-[#faedec] flex gap-[12px] items-start p-[16px] rounded-[6px] w-full">
            <CircleAlert size={20} className="text-[#9a4948] shrink-0" />
            <div className="flex flex-col gap-[3px]">
              <p className="font-semibold text-[#9a4948] text-[16px]">
                {missingColumns.columns.length} required column{missingColumns.columns.length > 1 ? "s" : ""} {missingColumns.columns.length > 1 ? "are" : "is"} missing
              </p>
              <p className="text-[#9a4948] text-[14px] leading-[1.4]">
                {missingColumns.fileName} is missing {joinWithAnd(missingColumns.columns)}. Add{" "}
                {missingColumns.columns.length === 1 ? "the column" : missingColumns.columns.length === 2 ? "both columns" : "all columns"}{" "}
                or use the template, then upload again. No predictions were generated.
              </p>
            </div>
          </div>
        )}

        <div className="bg-white border border-[#dde4df] rounded-[10px] p-[24px] flex flex-col gap-[18px] w-full">
          <div className="flex items-start justify-between w-full">
            <p className="flex-1 font-semibold text-[#23362f] text-[20px]">CSV column checklist</p>
            <p className="text-[#65736d] text-[14px]">19 required · client_id optional</p>
          </div>
          <div className="flex gap-[20px] items-start w-full">
            {COLUMN_GROUPS.map((group) => (
              <div key={group.title} className="flex flex-1 flex-col gap-[8px] items-start min-w-0">
                <p className="font-semibold text-[#23362f] text-[14px]">{group.title}</p>
                {group.columns.map((col) => (
                  <p key={col} className="font-mono text-[#65736d] text-[13px]">{col}</p>
                ))}
              </div>
            ))}
          </div>
          <p className="text-[#65736d] text-[13px]">
            Keep column names exactly as shown. Add client_id to identify customers; otherwise a row
            reference is displayed. Use pdays = 999 if there was no prior contact.
          </p>
        </div>

        {error && (
          <div className="bg-[#fef3f2] border border-[#fecdca] rounded-[8px] px-[16px] py-[10px] w-full text-[14px] text-[#b42318]">
            {error}
          </div>
        )}

        {rowErrors && (
          <div className="flex flex-col gap-[12px] items-start w-full">
            <div className="bg-[#faedec] flex gap-[12px] items-start p-[16px] rounded-[6px] w-full">
              <CircleAlert size={20} className="text-[#9a4948] shrink-0" />
              <div className="flex flex-col gap-[3px]">
                <p className="font-semibold text-[#9a4948] text-[16px]">
                  {rowErrors.total_errors ?? rowErrors.errors?.length ?? 0} row
                  {(rowErrors.total_errors ?? 0) > 1 ? "s" : ""} need correction
                </p>
                <p className="text-[#9a4948] text-[14px] leading-[1.4]">
                  {rowErrors.fileName} contains invalid values. Correct the rows below and upload the file again.
                  No customers have been ranked.
                </p>
              </div>
            </div>

            <div className="bg-white border border-[#dde4df] rounded-[10px] p-[24px] flex flex-col gap-[18px] w-full">
              <div className="flex items-start justify-between w-full">
                <p className="flex-1 font-semibold text-[#23362f] text-[20px]">Rows to correct</p>
                <button onClick={downloadErrorReport}
                  className="bg-white border border-[#dde4df] flex gap-[8px] items-center px-[18px] py-[10px] rounded-[6px]">
                  <Download size={20} className="text-[#176547]" />
                  <span className="font-semibold text-[16px] text-[#176547]">Download error report</span>
                </button>
              </div>
              <div className="flex flex-col items-start w-full">
                <div className="bg-[#eef1ef] flex font-semibold gap-[20px] items-start px-[14px] py-[10px] w-full text-[#65736d] text-[13px]">
                  <p className="w-[100px]">CSV row</p>
                  <p className="w-[160px]">Column</p>
                  <p className="w-[120px]">Value</p>
                  <p className="flex-1">How to fix</p>
                </div>
                {rowErrors.errors?.map((e, i) => (
                  <div key={i} className="border-b border-[#dde4df] flex gap-[20px] items-start p-[14px] w-full text-[15px]">
                    <p className="text-[#9a4948] w-[100px]">Row {e.row}</p>
                    <p className="font-mono text-[#23362f] w-[160px]">{e.field}</p>
                    <p className="text-[#9a4948] w-[120px] truncate">{e.value}</p>
                    <p className="text-[#23362f] flex-1">{explainFieldError(e.field)}</p>
                  </div>
                ))}
              </div>
              {rowErrors.total_errors && rowErrors.errors && rowErrors.total_errors > rowErrors.errors.length && (
                <p className="text-[#65736d] text-[13px]">
                  … and {rowErrors.total_errors - rowErrors.errors.length} more row(s) with errors — see the full
                  error report.
                </p>
              )}
              <p className="text-[#65736d] text-[13px]">
                CSV row numbers include the header as row 1. Each issue must be fixed before the list can be processed.
              </p>
            </div>
          </div>
        )}

        {response && (
          <div className="flex flex-col gap-[20px] items-start w-full">
            <div className="flex gap-[16px] items-start w-full">
              <div className="bg-white border border-[#dde4df] rounded-[10px] p-[24px] flex flex-1 flex-col gap-[18px]">
                <p className="text-[#65736d] text-[14px]">Customers uploaded</p>
                <p className="font-semibold text-[#23362f] text-[32px]">{response.count}</p>
                <p className="text-[#65736d] text-[14px]">All customer rows processed</p>
              </div>
              <div className="bg-white border border-[#dde4df] rounded-[10px] p-[24px] flex flex-1 flex-col gap-[18px]">
                <p className="text-[#65736d] text-[14px]">Marked High priority</p>
                <p className="font-semibold text-[#23362f] text-[32px]">{counts.High}</p>
                <p className="text-[#65736d] text-[14px]">{high != null ? `${high}% probability or higher` : "—"}</p>
              </div>
              <div className="bg-white border border-[#dde4df] rounded-[10px] p-[24px] flex flex-1 flex-col gap-[18px]">
                <p className="text-[#65736d] text-[14px]">Recommended to call</p>
                <p className="font-semibold text-[#23362f] text-[32px]">{counts.Call}</p>
                <p className="text-[#65736d] text-[14px]">{high != null ? `At or above the ${medium}% threshold` : "—"}</p>
              </div>
            </div>

            <div className="bg-white border border-[#dde4df] rounded-[10px] p-[24px] flex flex-col gap-[18px] w-full">
              <div className="flex items-center justify-between w-full">
                <div className="flex flex-col gap-[4px]">
                  <p className="font-semibold text-[#23362f] text-[22px]">Your prioritised call list</p>
                  <p className="text-[#65736d] text-[14px]">Highest probability first · {response.count} customers</p>
                </div>
                <button onClick={downloadRankedCsv}
                  className="bg-[#176547] border border-[#176547] flex gap-[8px] items-center px-[18px] py-[10px] rounded-[6px]">
                  <Download size={20} className="text-white" />
                  <span className="font-semibold text-[16px] text-white">Download ranked list</span>
                </button>
              </div>

              <div className="flex gap-[8px] items-center w-full">
                <p className="text-[#65736d] text-[14px]">Priority</p>
                {(["All", "High", "Medium", "Low"] as const).map((f) => (
                  <button key={f} onClick={() => setFilter(f)}
                    className={`px-[14px] py-[7px] rounded-full text-[14px] border ${
                      filter === f ? "bg-[#eaf3ee] border-[#176547] font-semibold text-[#176547]" : "bg-white border-[#dde4df] text-[#65736d]"
                    }`}>
                    {f}
                  </button>
                ))}
              </div>

              <div className="flex flex-col items-start w-full">
                <div className="bg-[#eef1ef] flex font-semibold h-[44px] items-center px-[12px] w-full text-[#65736d] text-[14px]">
                  <p className="w-[58px]">Rank</p>
                  <p className="w-[170px]">client_id</p>
                  <p className="w-[180px]">Probability</p>
                  <p className="w-[170px]">Predicted class</p>
                  <p className="w-[220px]">Recommendation</p>
                  <p className="w-[140px]">Priority</p>
                </div>
                {shown.map((row) => (
                  <div key={row.rank} className="bg-white border-b border-[#dde4df] flex h-[56px] items-center px-[12px] w-full">
                    <p className="text-[#65736d] text-[15px] w-[58px]">{row.rank}</p>
                    <p className="font-mono text-[#23362f] text-[14px] w-[170px] truncate">
                      {String(row.client_id ?? `Row ${row.rank}`)}
                    </p>
                    <div className="flex gap-[10px] items-center w-[180px]">
                      <p className="font-semibold text-[#23362f] text-[16px] w-[36px]">
                        {(row.probability * 100).toFixed(0)}%
                      </p>
                      <div className="bg-[#eef1ef] flex h-[5px] rounded-[3px] w-[72px] overflow-hidden">
                        <div className="bg-[#176547] h-[5px] rounded-[3px]" style={{ width: `${Math.min(row.probability * 100, 100)}%` }} />
                      </div>
                    </div>
                    <p className="text-[#23362f] text-[15px] w-[170px]">{row.predicted_class === "yes" ? "Yes" : "No"}</p>
                    <div className="w-[220px]">
                      <StatusPill bg={row.recommendation === "Call" ? "#eaf3ee" : "#eef1ef"}
                        text={row.recommendation === "Call" ? "#176547" : "#65736d"}>
                        {row.recommendation}
                      </StatusPill>
                    </div>
                    <div className="w-[140px]">
                      <StatusPill bg={PRIORITY_BADGE[row.priority].bg} text={PRIORITY_BADGE[row.priority].text}>
                        {row.priority}
                      </StatusPill>
                    </div>
                  </div>
                ))}
              </div>
              <p className="text-[#65736d] text-[13px]">
                {high != null && medium != null
                  ? `Showing ${shown.length} of ${response.count} customers · High ≥${high}%, Medium ${medium}–<${high}%, Low <${medium}%. Predicted class Yes and Call both use the ${high}% threshold.`
                  : `Showing ${shown.length} of ${response.count} customers.`}
              </p>
            </div>
          </div>
        )}

        {!response && !error && !rowErrors && !fileTypeError && !missingColumns && (
          <div className="flex gap-[14px] items-center p-[24px] w-full">
            <ListOrdered size={28} className="text-[#65736d] shrink-0" />
            <div className="flex flex-1 flex-col gap-[4px]">
              <p className="font-semibold text-[#23362f] text-[20px]">Your ranked list will appear here</p>
              <p className="text-[#65736d] text-[16px]">
                Upload a valid CSV to see recommendations, priority filters and a downloadable list.
              </p>
            </div>
          </div>
        )}

        <div className="border-[#dde4df] border-solid border-t flex gap-[10px] items-start py-[16px] w-full">
          <ShieldCheck size={18} className="text-[#65736d] shrink-0" />
          <p className="flex-1 text-[#65736d] text-[14px] leading-[1.4]">
            Decision support only. Check contact preferences and opt-outs before calling.
          </p>
        </div>
      </div>
    </div>
  );
}
