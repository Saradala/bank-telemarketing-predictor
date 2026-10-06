"use client";

import { ArrowRight, CircleAlert, CircleCheck, Info, ListOrdered, ShieldCheck, UserRound } from "lucide-react";
import Link from "next/link";
import { useEffect, useState } from "react";
import { fetchHealth, fetchModelInfo } from "@/lib/api";

type Status = {
  reachable: boolean;
  modelLoaded: boolean;
  modelName: string | null;
  threshold: number | null;
};

type Performance = {
  available: boolean;
  model?: string;
  test_pr_auc?: number;
  recall?: number;
  precision?: number;
};

function MetricCard({ label, value, caption }: { label: string; value: React.ReactNode; caption: string }) {
  return (
    <div className="bg-white border border-[#dde4df] border-solid flex flex-1 flex-col gap-[10px] p-[22px] rounded-[10px]">
      <p className="text-[#65736d] text-[14px]">{label}</p>
      <div className="flex items-center gap-[8px] w-full">{value}</div>
      <p className="text-[#65736d] text-[14px]">{caption}</p>
    </div>
  );
}

export default function Home() {
  const [status, setStatus] = useState<Status>({
    reachable: false, modelLoaded: false, modelName: null, threshold: null,
  });
  const [loadingStatus, setLoadingStatus] = useState(true);
  const [performance, setPerformance] = useState<Performance>({ available: false });
  const available = status.reachable && status.modelLoaded;

  useEffect(() => {
    (async () => {
      try {
        const health = await fetchHealth();
        if (health.model_loaded) {
          const info = await fetchModelInfo();
          setStatus({ reachable: true, modelLoaded: true, modelName: info.model_name, threshold: info.threshold });
        } else {
          setStatus({ reachable: true, modelLoaded: false, modelName: null, threshold: null });
        }
      } catch {
        setStatus({ reachable: false, modelLoaded: false, modelName: null, threshold: null });
      } finally {
        setLoadingStatus(false);
      }
    })();

    fetch("/api/model-performance")
      .then((res) => res.json())
      .then(setPerformance)
      .catch(() => setPerformance({ available: false }));
  }, []);

  return (
    <div className="flex flex-col items-start w-full">
      <div className="flex flex-col gap-[28px] items-start p-[40px] w-full max-w-[1200px] mx-auto">
        <div className="flex flex-col gap-[8px] items-start w-full">
          <div className="flex items-center justify-between w-full">
            <p className="font-semibold text-[#65736d] text-[13px] uppercase">Overview</p>
          </div>
          <p className="font-semibold text-[#23362f] text-[34px] leading-[1.4] w-full">
            Term Deposit Call Prioritisation
          </p>
          <p className="text-[#65736d] text-[16px] leading-[1.4] w-full">
            Find the customers most likely to subscribe, so your team can focus its calls.
          </p>
        </div>

        <div className="flex flex-col gap-[12px] items-start w-full">
          <div className="flex items-start justify-between w-full">
            <p className="flex-1 font-semibold text-[#23362f] text-[20px] leading-[1.4]">System status</p>
          </div>
          {!loadingStatus && !status.reachable && (
            <div className="bg-[#fbf3e2] flex gap-[12px] items-start p-[16px] rounded-[6px] w-full">
              <Info size={20} className="text-[#865a12] shrink-0" />
              <div className="flex flex-col gap-[3px]">
                <p className="font-semibold text-[#865a12] text-[16px]">Predictions are temporarily unavailable</p>
                <p className="text-[#865a12] text-[14px] leading-[1.4]">
                  The API is offline and no model is loaded. Your team can return when the service is
                  restored; no customer data has been submitted.
                </p>
              </div>
            </div>
          )}
          {!loadingStatus && status.reachable && !status.modelLoaded && (
            <div className="bg-[#fbf3e2] flex gap-[12px] items-start p-[16px] rounded-[6px] w-full">
              <Info size={20} className="text-[#865a12] shrink-0" />
              <div className="flex flex-col gap-[3px]">
                <p className="font-semibold text-[#865a12] text-[16px]">Predictions are temporarily unavailable</p>
                <p className="text-[#865a12] text-[14px] leading-[1.4]">
                  The API is running, but no model has been loaded yet. Your team can return shortly;
                  no customer data has been submitted.
                </p>
              </div>
            </div>
          )}
          <div className="flex gap-[16px] items-start w-full">
            <MetricCard
              label="API connection"
              value={<>
                {status.reachable
                  ? <CircleCheck size={20} className="text-[#176547]" />
                  : <CircleAlert size={20} className="text-[#865a12]" />}
                <p className="flex-1 font-semibold text-[#23362f] text-[30px] leading-[1.4]">
                  {loadingStatus ? "Checking…" : status.reachable ? "Online" : "Offline"}
                </p>
              </>}
              caption={status.reachable ? "Live connection status" : "Service cannot be reached"}
            />
            <MetricCard
              label="Loaded model"
              value={<p className="flex-1 font-semibold text-[#23362f] text-[24px] leading-[1.4]">
                {status.modelName ?? "Not loaded"}
              </p>}
              caption={status.modelName ? "Currently selected model" : "No model information available"}
            />
            <MetricCard
              label="Current call threshold"
              value={<p className="flex-1 font-semibold text-[#23362f] text-[30px] leading-[1.4]">
                {status.threshold != null ? `${Math.round(status.threshold * 100)}%` : "—"}
              </p>}
              caption={status.threshold != null
                ? `Call when probability is ${Math.round(status.threshold * 100)}% or higher`
                : "Available after a model is loaded"}
            />
          </div>
        </div>

        <div className="flex gap-[20px] items-start w-full">
          <div className="bg-white border border-[#dde4df] border-solid flex flex-1 flex-col gap-[18px] p-[24px] rounded-[10px]">
            <div className="flex gap-[12px] items-center w-full">
              <div className="bg-[#eaf3ee] flex items-center justify-center rounded-[8px] shrink-0 size-[40px]">
                <UserRound size={20} className="text-[#176547]" />
              </div>
              <p className="flex-1 font-semibold text-[#23362f] text-[23px] leading-[1.4]">Check one customer</p>
            </div>
            <p className="text-[#65736d] text-[16px] leading-[1.4]">
              Enter a customer&apos;s details and review a clear call recommendation.
            </p>
            <Link href="/predict"
              className={`flex gap-[8px] items-center px-[18px] py-[10px] rounded-[6px] border ${
                available ? "bg-[#176547] border-[#176547]" : "bg-[#eef1ef] border-[#dde4df]"
              }`}>
              <ArrowRight size={20} className={available ? "text-white" : "text-[#8a958f]"} />
              <span className={`font-semibold text-[16px] ${available ? "text-white" : "text-[#8a958f]"}`}>
                Check one customer
              </span>
            </Link>
          </div>
          <div className="bg-white border border-[#dde4df] border-solid flex flex-1 flex-col gap-[18px] p-[24px] rounded-[10px]">
            <div className="flex gap-[12px] items-center w-full">
              <div className="bg-[#eaf3ee] flex items-center justify-center rounded-[8px] shrink-0 size-[40px]">
                <ListOrdered size={20} className="text-[#176547]" />
              </div>
              <p className="flex-1 font-semibold text-[#23362f] text-[23px] leading-[1.4]">
                Build a prioritised call list
              </p>
            </div>
            <p className="text-[#65736d] text-[16px] leading-[1.4]">
              Upload your CSV to rank customers and plan your next round of calls.
            </p>
            <Link href="/batch"
              className={`flex gap-[8px] items-center px-[18px] py-[10px] rounded-[6px] border ${
                available ? "bg-white border-[#dde4df]" : "bg-[#eef1ef] border-[#dde4df]"
              }`}>
              <ArrowRight size={20} className={available ? "text-[#176547]" : "text-[#8a958f]"} />
              <span className={`font-semibold text-[16px] ${available ? "text-[#176547]" : "text-[#8a958f]"}`}>
                Upload a customer list
              </span>
            </Link>
          </div>
        </div>

        <div className="flex flex-col gap-[12px] items-start w-full">
          <div className="flex items-start justify-between w-full">
            <p className="flex-1 font-semibold text-[#23362f] text-[20px] leading-[1.4]">Model performance</p>
          </div>
          <p className="text-[#65736d] text-[14px] w-full">
            {performance.available
              ? `Test results for ${performance.model}, from the team's experiment log.`
              : "No experiment results found yet."}
          </p>
          {performance.available && (
            <div className="flex gap-[16px] items-start w-full">
              <MetricCard label="Best model · PR-AUC"
                value={<p className="font-semibold text-[#23362f] text-[30px] leading-[1.4]">{performance.test_pr_auc?.toFixed(2)}</p>}
                caption="How well the model finds subscribers across thresholds." />
              <MetricCard label="Recall"
                value={<p className="font-semibold text-[#23362f] text-[30px] leading-[1.4]">{performance.recall != null ? `${Math.round(performance.recall * 100)}%` : "—"}</p>}
                caption="Share of all subscribers the call recommendation finds." />
              <MetricCard label="Precision"
                value={<p className="font-semibold text-[#23362f] text-[30px] leading-[1.4]">{performance.precision != null ? `${Math.round(performance.precision * 100)}%` : "—"}</p>}
                caption="Share of recommended customers who subscribe." />
            </div>
          )}
        </div>

        <div className="bg-white border border-[#dde4df] border-solid flex flex-col gap-[18px] p-[24px] rounded-[10px] w-full">
          <p className="font-semibold text-[#23362f] text-[21px] leading-[1.4] w-full">How to use</p>
          <div className="flex gap-[24px] items-start w-full">
            {[
              { n: 1, title: "Choose your customers", body: "Check one customer or upload a CSV list." },
              { n: 2, title: "Review recommendations", body: "Start with High priority; check the probability." },
              { n: 3, title: "Call responsibly", body: "Review opt-outs, then download your call list." },
            ].map((step) => (
              <div key={step.n} className="flex flex-1 gap-[12px] items-start">
                <div className="bg-[#eaf3ee] flex items-center justify-center rounded-full shrink-0 size-[28px]">
                  <p className="font-semibold text-[#176547] text-[14px]">{step.n}</p>
                </div>
                <div className="flex flex-1 flex-col gap-[4px] items-start leading-[1.4]">
                  <p className="font-semibold text-[#23362f] text-[16px] w-full">{step.title}</p>
                  <p className="text-[#65736d] text-[14px] w-full">{step.body}</p>
                </div>
              </div>
            ))}
          </div>
        </div>

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
