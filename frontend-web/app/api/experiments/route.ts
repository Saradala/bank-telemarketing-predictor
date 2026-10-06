import { readFile } from "fs/promises";
import path from "path";
import { NextResponse } from "next/server";
import { parseCsv } from "@/lib/csv";

const LOG_PATH = path.join(process.cwd(), "..", "results", "experiment_log.csv");

export async function GET() {
  let raw: string;
  try {
    raw = await readFile(LOG_PATH, "utf-8");
  } catch {
    return NextResponse.json({ rows: [] });
  }
  return NextResponse.json({ rows: parseCsv(raw) });
}
