"use client";

import { ArrowRight, ChartNoAxesCombined, Info, ListOrdered, Repeat2, ShieldCheck, Smartphone } from "lucide-react";
import Link from "next/link";
import { useEffect, useState } from "react";
import { fetchModelInfo, ModelInfoResponse } from "@/lib/api";

type ExperimentRow = Record<string, string>;

const FAMILY_LABELS: Record<string, string> = {
  LogisticRegression: "Logistic Regression",
  XGBoost: "XGBoost",
  "Random Forest": "Random Forest",
  SVM: "SVM",
};

function familyOf(model: string): string {
  return model.split(/\s*[-(]/)[0].trim();
}

function bestPerFamily(rows: ExperimentRow[]): { family: string; row: ExperimentRow }[] {
  const comparable = rows.filter((r) => r.test_pr_auc && !r.notes?.toLowerCase().includes("not comparable"));
  const byFamily = new Map<string, ExperimentRow>();
  for (const row of comparable) {
    const family = familyOf(row.model);
    const current = byFamily.get(family);
    if (!current || parseFloat(row.test_pr_auc) > parseFloat(current.test_pr_auc)) byFamily.set(family, row);
  }
  return [...byFamily.entries()]
    .map(([family, row]) => ({ family, row }))
    .sort((a, b) => parseFloat(b.row.test_pr_auc) - parseFloat(a.row.test_pr_auc));
}

function formatMetric(value: string): string {
  if (!value) return "—";
  const n = parseFloat(value);
  return Number.isNaN(n) ? value : n.toFixed(4);
}

function FindingCard({ stat, icon, title, body }: { stat: string; icon: React.ReactNode; title: string; body: string }) {
  return (
    <div className="bg-white border border-[#dde4df] rounded-[10px] p-[24px] flex flex-1 flex-col gap-[18px] min-w-0">
      <div className="flex items-center justify-between w-full">
        <p className="font-semibold text-[#176547] text-[44px] leading-[1.1]">{stat}</p>
        {icon}
      </div>
      <p className="font-semibold text-[#23362f] text-[18px] leading-[1.4]">{title}</p>
      <p className="text-[#65736d] text-[14px] leading-[1.4]">{body}</p>
      <span className="flex gap-[6px] items-center px-[10px] py-[4px] rounded-full bg-[#eef1ef] w-fit">
        <span className="size-[5px] rounded-full bg-[#65736d]" />
        <span className="font-semibold text-[13px] text-[#65736d]">Example finding—not validated</span>
      </span>
    </div>
  );
}

export default function InsightsPage() {
  const [rows, setRows] = useState<ExperimentRow[] | null>(null);
  const [modelInfo, setModelInfo] = useState<ModelInfoResponse | null>(null);
  const [showAll, setShowAll] = useState(false);

  useEffect(() => {
    fetch("/api/experiments").then((r) => r.json()).then((d) => setRows(d.rows));
    fetchModelInfo().then(setModelInfo).catch(() => setModelInfo(null));
  }, []);

  const families = rows ? bestPerFamily(rows) : [];
  const maxScore = families.length ? parseFloat(families[0].row.test_pr_auc) : 0;
  const axisMax = Math.ceil(maxScore * 10) / 10 + 0.1;
  const ticks = Array.from({ length: 6 }, (_, i) => (axisMax * i) / 5);

  const chosenFamily = modelInfo ? families.find((f) => FAMILY_LABELS[f.family] === modelInfo.model_name) : undefined;

  const loaded = rows !== null;
  const hasData = loaded && families.length > 0;

  return (
    <div className="flex flex-col items-start w-full">
      <div className="flex flex-col gap-[28px] items-start p-[40px] w-full max-w-[1200px] mx-auto">
        <div className="flex flex-col gap-[8px] items-start w-full">
          <p className="font-semibold text-[#65736d] text-[13px] uppercase">Insights / Model comparison</p>
          <p className="font-semibold text-[#23362f] text-[34px] leading-[1.4]">Understand the recommendations</p>
          <p className="text-[#65736d] text-[16px] leading-[1.4]">
            See how the selected model compares and what the results could mean for your team.
          </p>
        </div>

        {hasData ? (
          <div className="bg-[#eef1ef] flex gap-[12px] items-start p-[16px] rounded-[6px] w-full">
            <Info size={20} className="text-[#65736d] shrink-0" />
            <div className="flex flex-1 flex-col gap-[3px]">
              <p className="font-semibold text-[#65736d] text-[16px]">Scores from the team&apos;s own evaluation</p>
              <p className="text-[#65736d] text-[14px] leading-[1.4]">
                These are the held-out test-set results logged during model development, not measured outcomes
                from a live calling campaign.
              </p>
            </div>
          </div>
        ) : loaded && (
          <div className="bg-[#eef1ef] flex gap-[12px] items-start p-[16px] rounded-[6px] w-full">
            <Info size={20} className="text-[#65736d] shrink-0" />
            <div className="flex flex-1 flex-col gap-[3px]">
              <p className="font-semibold text-[#65736d] text-[16px]">No model evaluation has been loaded</p>
              <p className="text-[#65736d] text-[14px] leading-[1.4]">
                There are no test scores, selected-model details or validated findings to display yet.
                Customer predictions should not be used until a model is available.
              </p>
            </div>
          </div>
        )}

        {hasData ? (
          <div className="flex gap-[24px] items-start w-full">
            <div className="bg-white border border-[#dde4df] rounded-[10px] p-[24px] flex flex-col gap-[18px] w-[688px] shrink-0">
              <p className="font-semibold text-[#23362f] text-[21px]">How the models compare</p>
              <p className="text-[#65736d] text-[14px] -mt-[8px]">Test PR-AUC · higher is better</p>

              <div className="flex flex-col gap-[22px] items-start pb-[8px] pt-[12px] w-full">
                {families.map(({ family, row }) => {
                  const selected = FAMILY_LABELS[family] === modelInfo?.model_name;
                  const score = parseFloat(row.test_pr_auc);
                  return (
                    <div key={family} className="flex gap-[16px] items-center w-full">
                      <p className={`text-[15px] w-[155px] shrink-0 ${selected ? "font-semibold text-[#176547]" : "text-[#65736d]"}`}>
                        {FAMILY_LABELS[family] ?? family}
                      </p>
                      <div className="bg-[#f4f6f5] flex h-[34px] rounded-[4px] w-full overflow-hidden">
                        <div className="h-[34px] rounded-[4px]"
                          style={{ width: `${(score / axisMax) * 100}%`, backgroundColor: selected ? "#176547" : "#b9c7be" }} />
                      </div>
                      <p className={`text-[16px] shrink-0 w-[40px] text-right font-semibold ${selected ? "text-[#176547]" : "text-[#23362f]"}`}>
                        {score.toFixed(2)}
                      </p>
                    </div>
                  );
                })}
                <div className="flex gap-[16px] items-start w-full">
                  <div className="w-[155px] shrink-0" />
                  <div className="flex justify-between w-full text-[#65736d] text-[11px]">
                    {ticks.map((t, i) => <p key={i}>{t.toFixed(2)}</p>)}
                  </div>
                  <div className="w-[40px] shrink-0" />
                </div>
              </div>
              <p className="text-[#65736d] text-[14px]">
                PR-AUC summarises how well a model finds subscribers while limiting unnecessary calls,
                across different thresholds.
              </p>
            </div>

            <div className="bg-white border border-[#dde4df] rounded-[10px] p-[24px] flex flex-1 flex-col gap-[18px] min-w-0">
              <span className="flex gap-[6px] items-center px-[10px] py-[4px] rounded-full bg-[#eaf3ee] w-fit">
                <span className="size-[5px] rounded-full bg-[#176547]" />
                <span className="font-semibold text-[13px] text-[#176547]">Deployed model</span>
              </span>
              <p className="text-[#65736d] text-[14px]">Final model</p>
              <p className="font-semibold text-[#23362f] text-[28px]">{modelInfo?.model_name ?? "Not loaded"}</p>
              <p className="font-semibold text-[#23362f] text-[16px]">Why this model?</p>
              <p className="text-[#65736d] text-[16px] leading-[1.4]">
                It has the highest test PR-AUC among the models the team compared, and as a linear model it is
                simpler to explain and defend than the tree-based alternatives.
              </p>
              <div className="bg-[#f4f6f5] flex flex-col gap-[8px] p-[16px] rounded-[6px] w-full text-[#23362f] text-[14px]">
                <p>Test PR-AUC: {chosenFamily ? parseFloat(chosenFamily.row.test_pr_auc).toFixed(2) : "—"}</p>
                <p>Deployed call threshold: {modelInfo ? `${Math.round(modelInfo.threshold * 100)}%` : "—"}</p>
                <p>
                  Recall: {chosenFamily ? `${Math.round(parseFloat(chosenFamily.row.recall) * 100)}%` : "—"} ·
                  {" "}Precision: {chosenFamily ? `${Math.round(parseFloat(chosenFamily.row.precision) * 100)}%` : "—"}
                  {" "}<span className="text-[#65736d]">(at the 50% reference threshold logged for this test)</span>
                </p>
              </div>
              <p className="text-[#65736d] text-[13px]">
                Selection and threshold must be confirmed using your bank&apos;s own evaluation and contact policy.
              </p>
            </div>
          </div>
        ) : loaded && (
          <>
            <div className="bg-white border border-[#dde4df] rounded-[10px] flex flex-col items-start w-full">
              <div className="flex flex-col gap-[16px] items-center px-[150px] py-[52px] w-full">
                <div className="bg-[#eaf3ee] flex items-center justify-center rounded-full shrink-0 size-[64px]">
                  <ChartNoAxesCombined size={30} className="text-[#176547]" />
                </div>
                <p className="font-semibold text-[#23362f] text-[26px] text-center">No comparison results yet</p>
                <p className="text-[#65736d] text-[16px] leading-[1.5] text-center">
                  When model evaluation is available, you&apos;ll see four model scores, the selected model and
                  plain-language insights here.
                </p>
                <Link href="/"
                  className="bg-white border border-[#dde4df] flex gap-[8px] items-center px-[18px] py-[10px] rounded-[6px]">
                  <ArrowRight size={20} className="text-[#176547]" />
                  <span className="font-semibold text-[16px] text-[#176547]">Check system status</span>
                </Link>
              </div>
            </div>

            <div className="flex gap-[20px] items-start w-full">
              <div className="bg-white border border-[#dde4df] rounded-[10px] p-[24px] flex flex-1 flex-col gap-[18px] min-w-0">
                <p className="font-semibold text-[#23362f] text-[21px]">Model comparison</p>
                <p className="text-[#65736d] text-[16px]">
                  Awaiting test scores. No chart is shown until evaluation data is loaded.
                </p>
              </div>
              <div className="bg-white border border-[#dde4df] rounded-[10px] p-[24px] flex flex-1 flex-col gap-[18px] min-w-0">
                <p className="font-semibold text-[#23362f] text-[21px]">Selected model</p>
                <p className="text-[#65736d] text-[16px]">
                  Not available. The final model and reason for selection will appear here.
                </p>
              </div>
              <div className="bg-white border border-[#dde4df] rounded-[10px] p-[24px] flex flex-1 flex-col gap-[18px] min-w-0">
                <p className="font-semibold text-[#23362f] text-[21px]">Outreach insights</p>
                <p className="text-[#65736d] text-[16px]">
                  No validated findings available. Review new findings before applying them to a campaign.
                </p>
              </div>
            </div>
          </>
        )}

        {hasData && (
          <div className="flex flex-col gap-[14px] items-start w-full">
            <p className="font-semibold text-[#23362f] text-[23px]">What this could mean for outreach</p>
            <div className="flex gap-[16px] items-start w-full">
              <FindingCard stat="6×" icon={<Repeat2 size={24} className="text-[#176547]" />}
                title="Customers who said yes before are 6x more likely to say yes again"
                body="Prior campaign response may help your team decide who to review first." />
              <FindingCard stat="3×" icon={<Smartphone size={24} className="text-[#176547]" />}
                title="Mobile contacts convert 3x more often than landline"
                body="Review preferred contact channels; never assume consent from a score." />
              <FindingCard stat="37%" icon={<ListOrdered size={24} className="text-[#176547]" />}
                title="Calling the top 20% of the ranked list reaches 37% of all subscribers"
                body="A focused list may help teams use limited calling time more effectively." />
            </div>
          </div>
        )}

        <div className="bg-white border border-[#dde4df] rounded-[10px] p-[24px] flex flex-col gap-[18px] w-full">
          <p className="font-semibold text-[#23362f] text-[21px]">A score supports a decision. It doesn&apos;t replace one.</p>
          <p className="text-[#65736d] text-[16px]">
            Model scores describe patterns, not promises about individual customers. Confirm your campaign
            policy and contact permissions before acting on any recommendation.
          </p>
        </div>

        {rows && rows.length > 0 && (
        <div className="w-full">
          <button onClick={() => setShowAll((v) => !v)} className="font-semibold text-[14px] text-[#176547]">
            {showAll ? "Hide" : "Show"} every logged experiment
          </button>
          {showAll && (
            <div className="bg-white border border-[#dde4df] rounded-[10px] overflow-x-auto w-full mt-[12px]">
              <table className="w-full text-[14px]">
                <thead>
                  <tr className="border-b border-[#dde4df] text-left text-[#65736d]">
                    <th className="p-[12px]">Model</th>
                    <th className="p-[12px]">CV PR-AUC</th>
                    <th className="p-[12px]">Test PR-AUC</th>
                    <th className="p-[12px]">Test ROC-AUC</th>
                    <th className="p-[12px]">Recall</th>
                    <th className="p-[12px]">Precision</th>
                    <th className="p-[12px]">F1</th>
                    <th className="p-[12px]">Notes</th>
                  </tr>
                </thead>
                <tbody>
                  {rows?.map((row, i) => (
                    <tr key={i} className="border-b border-[#f0f2f1] last:border-0 align-top">
                      <td className="p-[12px] font-semibold text-[#23362f] whitespace-nowrap">{row.model}</td>
                      <td className="p-[12px] text-[#23362f]">{formatMetric(row.cv_pr_auc)}</td>
                      <td className="p-[12px] text-[#23362f]">{formatMetric(row.test_pr_auc)}</td>
                      <td className="p-[12px] text-[#23362f]">{formatMetric(row.test_roc_auc)}</td>
                      <td className="p-[12px] text-[#23362f]">{formatMetric(row.recall)}</td>
                      <td className="p-[12px] text-[#23362f]">{formatMetric(row.precision)}</td>
                      <td className="p-[12px] text-[#23362f]">{formatMetric(row.f1)}</td>
                      <td className="p-[12px] text-[#65736d] min-w-[260px]">{row.notes}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
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
