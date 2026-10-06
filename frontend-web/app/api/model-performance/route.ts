import { readFile } from "fs/promises";
import path from "path";
import { NextResponse } from "next/server";
import { parseCsv } from "@/lib/csv";

// results/experiment_log.csv is the team's shared experiment log (not part of this app),
// one directory up from frontend-web/. Read server-side so the browser never needs a path
// into the repo, and so this works the same way the Streamlit dashboard's pd.read_csv() does.
const LOG_PATH = path.join(process.cwd(), "..", "results", "experiment_log.csv");

export async function GET() {
  let raw: string;
  try {
    raw = await readFile(LOG_PATH, "utf-8");
  } catch {
    return NextResponse.json({ available: false });
  }

  const rows = parseCsv(raw);
  const withTestScore = rows.filter((row) => row.test_pr_auc && row.test_pr_auc !== "");
  if (withTestScore.length === 0) return NextResponse.json({ available: false });

  const best = withTestScore.reduce((a, b) =>
    parseFloat(b.test_pr_auc) > parseFloat(a.test_pr_auc) ? b : a
  );

  return NextResponse.json({
    available: true,
    model: best.model,
    test_pr_auc: parseFloat(best.test_pr_auc),
    recall: parseFloat(best.recall),
    precision: parseFloat(best.precision),
  });
}
