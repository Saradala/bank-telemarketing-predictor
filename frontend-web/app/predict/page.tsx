"use client";

import { ChevronDown, CircleAlert, LoaderCircle, Phone, ShieldCheck, UserRoundSearch } from "lucide-react";
import { useEffect, useState } from "react";
import {
  ApiError, ClientFeatures, PredictionResponse, fetchModelInfo, predict,
} from "@/lib/api";
import {
  CONTACT_OPTIONS, EDUCATION_OPTIONS, JOB_OPTIONS, MARITAL_OPTIONS, MONTH_OPTIONS,
  POUTCOME_OPTIONS, PRIORITY_BADGE, WEEKDAY_OPTIONS, YES_NO_UNKNOWN_OPTIONS,
} from "@/lib/constants";

// Customer-specific fields start empty so the form matches the "empty form" state exactly;
// only the economic-context fields (not customer-specific) start pre-filled with example values.
type FormState = {
  age: number | ""; job: string; marital: string; education: string; default: string;
  housing: string; loan: string; contact: string; month: string; day_of_week: string;
  campaign: number | ""; pdays: number | ""; previous: number | ""; poutcome: string;
  emp_var_rate: number; cons_price_idx: number; cons_conf_idx: number; euribor3m: number; nr_employed: number;
};

const EMPTY_FORM: FormState = {
  age: "", job: "", marital: "", education: "", default: "", housing: "", loan: "", contact: "",
  month: "", day_of_week: "", campaign: "", pdays: "", previous: "", poutcome: "",
  emp_var_rate: -1.8, cons_price_idx: 92.893, cons_conf_idx: -46.2, euribor3m: 1.313, nr_employed: 5099.1,
};

const REQUIRED_KEYS: (keyof FormState)[] = [
  "age", "job", "marital", "education", "default", "housing", "loan", "contact", "month",
  "day_of_week", "campaign", "pdays", "previous", "poutcome",
];

// Matches the numeric bounds enforced by backend/schemas.py ClientFeatures, so a client-side
// rejection here always means the API would reject it too.
const RANGE_RULES: { key: "age" | "campaign" | "pdays" | "previous"; label: string; min: number; max: number }[] = [
  { key: "age", label: "Age", min: 18, max: 100 },
  { key: "campaign", label: "Contacts this campaign", min: 1, max: 60 },
  { key: "pdays", label: "Days since prior contact", min: 0, max: 999 },
  { key: "previous", label: "Prior campaign contacts", min: 0, max: 20 },
];

type FieldError = { label: string; message: string; caption: string };

function validateForm(form: FormState): Partial<Record<string, FieldError>> {
  const errors: Partial<Record<string, FieldError>> = {};
  for (const { key, label, min, max } of RANGE_RULES) {
    const value = form[key];
    if (value !== "" && (value < min || value > max)) {
      const noun = key === "age" ? "an age" : "a value";
      errors[key] = {
        label,
        message: `${label} must be between ${min} and ${max}.`,
        caption: `${key} · Enter ${noun} between ${min} and ${max}.`,
      };
    }
  }
  return errors;
}

const JOB_LABELS: Record<string, string> = { "admin.": "Administrative", "blue-collar": "Blue-collar" };
const LABEL_OVERRIDES: Record<string, string> = {
  university: "University degree", "high.school": "High school", "basic.4y": "Basic (4y)",
  "basic.6y": "Basic (6y)", "basic.9y": "Basic (9y)", "professional.course": "Professional course",
};

function titleCase(value: string): string {
  if (JOB_LABELS[value]) return JOB_LABELS[value];
  if (value === "university.degree") return "University degree";
  if (LABEL_OVERRIDES[value]) return LABEL_OVERRIDES[value];
  return value.charAt(0).toUpperCase() + value.slice(1);
}

function FieldWrap({ label, caption, error, children }: {
  label: string; caption: string; error?: string; children: React.ReactNode;
}) {
  return (
    <div className="flex flex-1 flex-col gap-[6px] min-w-0">
      <p className="font-semibold text-[#23362f] text-[15px]">{label} *</p>
      {children}
      <p className={`text-[12px] leading-[1.3] ${error ? "text-[#9a4948]" : "text-[#65736d]"}`}>
        {error ?? caption}
      </p>
    </div>
  );
}

function TextInput({ value, onChange, step, placeholder, invalid }: {
  value: number | ""; onChange: (v: number | "") => void; step?: number; placeholder?: string; invalid?: boolean;
}) {
  return (
    <input type="number" value={value} step={step ?? 1} placeholder={placeholder}
      onChange={(e) => onChange(e.target.value === "" ? "" : Number(e.target.value))}
      className={`h-[42px] rounded-[6px] px-[12px] text-[16px] bg-white w-full placeholder:text-[#8a958f] ${
        invalid ? "border-[1.5px] border-[#9a4948] text-[#9a4948]" : "border border-[#dde4df] text-[#23362f]"
      }`} />
  );
}

function SelectInput({ value, options, onChange }: { value: string; options: string[]; onChange: (v: string) => void }) {
  return (
    <div className="relative">
      <select value={value} onChange={(e) => onChange(e.target.value)}
        className={`h-[42px] border border-[#dde4df] rounded-[6px] pl-[12px] pr-[36px] text-[16px] bg-white w-full appearance-none ${value ? "text-[#23362f]" : "text-[#8a958f]"}`}>
        <option value="" disabled hidden>Select an option</option>
        {options.map((o) => <option key={o} value={o}>{titleCase(o)}</option>)}
      </select>
      <ChevronDown size={16} className="absolute right-[12px] top-1/2 -translate-y-1/2 text-[#65736d] pointer-events-none" />
    </div>
  );
}

function Section({ title, caption, children }: { title: string; caption?: string; children: React.ReactNode }) {
  return (
    <div className="bg-white border border-[#dde4df] rounded-[10px] p-[24px] flex flex-col gap-[18px] w-full">
      <p className="font-semibold text-[#23362f] text-[21px]">{title}</p>
      {caption && <p className="text-[#65736d] text-[14px] -mt-[8px]">{caption}</p>}
      {children}
    </div>
  );
}

function Row({ children }: { children: React.ReactNode }) {
  return <div className="flex gap-[16px] items-start w-full">{children}</div>;
}

export default function PredictPage() {
  const [form, setForm] = useState<FormState>(EMPTY_FORM);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [threshold, setThreshold] = useState<number | null>(null);
  const [apiReady, setApiReady] = useState<"checking" | "ready" | "unreachable" | "no-model">("checking");
  const [showErrors, setShowErrors] = useState(false);

  useEffect(() => {
    fetchModelInfo()
      .then((info) => { setThreshold(info.threshold); setApiReady("ready"); })
      .catch((err) => setApiReady(err instanceof ApiError && err.status === 503 ? "no-model" : "unreachable"));
  }, []);

  const set = <K extends keyof FormState>(key: K, value: FormState[K]) =>
    setForm((f) => ({ ...f, [key]: value }));

  const isComplete = REQUIRED_KEYS.every((key) => form[key] !== "");
  const fieldErrors = validateForm(form);
  const errorKeys = Object.keys(fieldErrors);

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!isComplete) return;
    if (errorKeys.length > 0) { setShowErrors(true); return; }
    setShowErrors(false);
    setSubmitting(true);
    setError(null);
    try {
      setResult(await predict(form as ClientFeatures));
    } catch (err) {
      setResult(null);
      setError(err instanceof ApiError
        ? (typeof err.detail === "string" ? err.detail : JSON.stringify(err.detail))
        : "Could not reach the API.");
    } finally {
      setSubmitting(false);
    }
  }

  function onClear() {
    setForm((f) => ({
      ...EMPTY_FORM,
      emp_var_rate: f.emp_var_rate, cons_price_idx: f.cons_price_idx,
      cons_conf_idx: f.cons_conf_idx, euribor3m: f.euribor3m, nr_employed: f.nr_employed,
    }));
    setResult(null);
    setError(null);
    setShowErrors(false);
  }

  const high = threshold != null ? Math.round(threshold * 100) : null;
  const medium = threshold != null ? Math.round((threshold / 2) * 100) : null;

  return (
    <div className="flex flex-col items-start w-full">
      <div className="flex flex-col gap-[28px] items-start p-[40px] w-full max-w-[1200px] mx-auto">
        <div className="flex flex-col gap-[8px] items-start w-full">
          <p className="font-semibold text-[#65736d] text-[13px] uppercase">Single prediction</p>
          <p className="font-semibold text-[#23362f] text-[34px] leading-[1.4]">Check one customer</p>
          <p className="text-[#65736d] text-[16px] leading-[1.4]">
            Enter the customer&apos;s details to see their likelihood of subscribing and a suggested next step.
          </p>
        </div>

        {apiReady === "unreachable" && (
          <div className="bg-[#fef3f2] border border-[#fecdca] rounded-[8px] px-[16px] py-[10px] w-full text-[14px] text-[#b42318]">
            Prediction API not reachable. Start it with: <code>uvicorn backend.app:app --reload --port 8001</code>
          </div>
        )}
        {apiReady === "no-model" && (
          <div className="bg-[#fffaeb] border border-[#fedf89] rounded-[8px] px-[16px] py-[10px] w-full text-[14px] text-[#b54708]">
            The API is running, but no model has been loaded yet.
          </div>
        )}

        {showErrors && errorKeys.length > 0 && (
          <div className="bg-[#faedec] flex gap-[12px] items-start p-[16px] rounded-[6px] w-full">
            <CircleAlert size={20} className="text-[#9a4948] shrink-0" />
            <div className="flex flex-col gap-[3px]">
              <p className="font-semibold text-[#9a4948] text-[16px]">
                Please correct {errorKeys.length} field{errorKeys.length > 1 ? "s" : ""} before predicting
              </p>
              <p className="text-[#9a4948] text-[14px] leading-[1.4]">
                {errorKeys.map((k) => fieldErrors[k]!.message).join(" ")} We haven&apos;t submitted this customer&apos;s details.
              </p>
            </div>
          </div>
        )}

        <div className="flex gap-[24px] items-start w-full">
          <form onSubmit={onSubmit} className="flex flex-col gap-[16px] items-start w-[788px] shrink-0">
            <p className="text-[#65736d] text-[14px]">
              All 19 fields are required. Technical field names are shown below each input.
            </p>

            <Section title="Client profile">
              <Row>
                <FieldWrap label="Age" caption="age · 18–100 years" error={showErrors ? fieldErrors.age?.caption : undefined}>
                  <TextInput value={form.age} placeholder="e.g. 42" invalid={showErrors && !!fieldErrors.age}
                    onChange={(v) => set("age", v)} />
                </FieldWrap>
                <FieldWrap label="Job" caption="job">
                  <SelectInput value={form.job} options={JOB_OPTIONS} onChange={(v) => set("job", v)} />
                </FieldWrap>
              </Row>
              <Row>
                <FieldWrap label="Marital status" caption="marital">
                  <SelectInput value={form.marital} options={MARITAL_OPTIONS} onChange={(v) => set("marital", v)} />
                </FieldWrap>
                <FieldWrap label="Education" caption="education">
                  <SelectInput value={form.education} options={EDUCATION_OPTIONS} onChange={(v) => set("education", v)} />
                </FieldWrap>
              </Row>
            </Section>

            <Section title="Credit and loans">
              <Row>
                <FieldWrap label="Credit in default?" caption="default · Yes / No / Unknown">
                  <SelectInput value={form.default} options={YES_NO_UNKNOWN_OPTIONS} onChange={(v) => set("default", v)} />
                </FieldWrap>
                <FieldWrap label="Housing loan?" caption="housing · Yes / No / Unknown">
                  <SelectInput value={form.housing} options={YES_NO_UNKNOWN_OPTIONS} onChange={(v) => set("housing", v)} />
                </FieldWrap>
                <FieldWrap label="Personal loan?" caption="loan · Yes / No / Unknown">
                  <SelectInput value={form.loan} options={YES_NO_UNKNOWN_OPTIONS} onChange={(v) => set("loan", v)} />
                </FieldWrap>
              </Row>
            </Section>

            <Section title="Contact details">
              <Row>
                <FieldWrap label="Contact channel" caption="contact · Cellular / Telephone">
                  <SelectInput value={form.contact} options={CONTACT_OPTIONS} onChange={(v) => set("contact", v)} />
                </FieldWrap>
                <FieldWrap label="Contact month" caption="month · January–December">
                  <SelectInput value={form.month} options={MONTH_OPTIONS} onChange={(v) => set("month", v)} />
                </FieldWrap>
                <FieldWrap label="Day of week" caption="day_of_week · Monday–Friday">
                  <SelectInput value={form.day_of_week} options={WEEKDAY_OPTIONS} onChange={(v) => set("day_of_week", v)} />
                </FieldWrap>
              </Row>
            </Section>

            <Section title="Campaign history">
              <Row>
                <FieldWrap label="Contacts this campaign" caption="campaign · Including this contact"
                  error={showErrors ? fieldErrors.campaign?.caption : undefined}>
                  <TextInput value={form.campaign} placeholder="e.g. 2" invalid={showErrors && !!fieldErrors.campaign}
                    onChange={(v) => set("campaign", v)} />
                </FieldWrap>
                <FieldWrap label="Days since prior contact" caption="pdays · 999 = not previously contacted"
                  error={showErrors ? fieldErrors.pdays?.caption : undefined}>
                  <TextInput value={form.pdays} placeholder="e.g. 999" invalid={showErrors && !!fieldErrors.pdays}
                    onChange={(v) => set("pdays", v)} />
                </FieldWrap>
              </Row>
              <Row>
                <FieldWrap label="Prior campaign contacts" caption="previous · Before this campaign"
                  error={showErrors ? fieldErrors.previous?.caption : undefined}>
                  <TextInput value={form.previous} placeholder="e.g. 0" invalid={showErrors && !!fieldErrors.previous}
                    onChange={(v) => set("previous", v)} />
                </FieldWrap>
                <FieldWrap label="Previous campaign outcome" caption="poutcome · Success / Failure / Nonexistent">
                  <SelectInput value={form.poutcome} options={POUTCOME_OPTIONS} onChange={(v) => set("poutcome", v)} />
                </FieldWrap>
              </Row>
            </Section>

            <Section title="Economic context" caption="Example defaults, not live economic figures. Edit these if your campaign uses different values.">
              <Row>
                <FieldWrap label="Employment variation rate" caption="emp_var_rate">
                  <TextInput value={form.emp_var_rate} step={0.1} onChange={(v) => set("emp_var_rate", v === "" ? 0 : v)} />
                </FieldWrap>
                <FieldWrap label="Consumer price index" caption="cons_price_idx">
                  <TextInput value={form.cons_price_idx} step={0.001} onChange={(v) => set("cons_price_idx", v === "" ? 0 : v)} />
                </FieldWrap>
                <FieldWrap label="Consumer confidence index" caption="cons_conf_idx">
                  <TextInput value={form.cons_conf_idx} step={0.1} onChange={(v) => set("cons_conf_idx", v === "" ? 0 : v)} />
                </FieldWrap>
              </Row>
              <Row>
                <FieldWrap label="3-month Euribor rate" caption="euribor3m">
                  <TextInput value={form.euribor3m} step={0.001} onChange={(v) => set("euribor3m", v === "" ? 0 : v)} />
                </FieldWrap>
                <FieldWrap label="Number employed" caption="nr_employed">
                  <TextInput value={form.nr_employed} step={0.1} onChange={(v) => set("nr_employed", v === "" ? 0 : v)} />
                </FieldWrap>
              </Row>
            </Section>

            <div className="flex items-center justify-between w-full">
              <button type="button" onClick={onClear} disabled={submitting}
                className={`border border-[#dde4df] rounded-[6px] px-[18px] py-[10px] font-semibold text-[16px] ${
                  submitting ? "bg-[#eef1ef] text-[#8a958f]" : "bg-white text-[#176547]"
                }`}>
                Clear customer details
              </button>
              <button type="submit" disabled={submitting || apiReady !== "ready" || !isComplete}
                className={`flex gap-[8px] items-center border rounded-[6px] px-[18px] py-[10px] font-semibold text-[16px] ${
                  submitting
                    ? "bg-[#eef1ef] border-[#dde4df] text-[#8a958f]"
                    : "bg-[#176547] border-[#176547] text-white disabled:opacity-60"
                }`}>
                {submitting && <LoaderCircle size={20} className="animate-spin" />}
                {submitting ? "Predicting…" : "Predict"}
              </button>
            </div>
            <p className="text-[#65736d] text-[13px]">
              Clearing customer details keeps the editable example economic defaults.
            </p>
          </form>

          <div className="flex flex-1 flex-col gap-[16px] items-start min-w-0">
            <div className="bg-white border border-[#dde4df] rounded-[10px] p-[24px] flex flex-col gap-[18px] w-full">
              <p className="font-semibold text-[#23362f] text-[20px]">Customer result</p>

              {error && !submitting && (
                <div className="bg-[#fef3f2] border border-[#fecdca] rounded-[8px] px-[16px] py-[10px] w-full text-[14px] text-[#b42318]">
                  {error}
                </div>
              )}

              {submitting && (
                <div className="flex flex-col gap-[14px] items-center py-[32px] w-full">
                  <div className="bg-[#eaf3ee] flex items-center justify-center rounded-full shrink-0 size-[56px]">
                    <LoaderCircle size={28} className="text-[#176547] animate-spin" />
                  </div>
                  <p className="font-semibold text-[#23362f] text-[20px] text-center">Checking this customer…</p>
                  <p className="text-[#65736d] text-[15px] text-center">
                    Please wait while the prediction is prepared. No recommendation is available yet.
                  </p>
                  <p className="text-[#176547] text-[13px]">Predicting…</p>
                </div>
              )}

              {!submitting && !result && !error && showErrors && errorKeys.length > 0 && (
                <div className="flex flex-col gap-[14px] items-center py-[32px] w-full">
                  <div className="bg-[#eaf3ee] flex items-center justify-center rounded-full shrink-0 size-[56px]">
                    <UserRoundSearch size={28} className="text-[#176547]" />
                  </div>
                  <p className="font-semibold text-[#23362f] text-[20px] text-center">
                    Correct the {errorKeys.length === 1 ? fieldErrors[errorKeys[0]]!.label.toLowerCase() : `${errorKeys.length} fields`} to continue
                  </p>
                  <p className="text-[#65736d] text-[15px] text-center">
                    We haven&apos;t submitted this customer. Fix the highlighted field{errorKeys.length > 1 ? "s" : ""}, then select Predict.
                  </p>
                </div>
              )}

              {!submitting && !result && !error && !(showErrors && errorKeys.length > 0) && (
                <div className="flex flex-col gap-[14px] items-center py-[32px] w-full">
                  <div className="bg-[#eaf3ee] flex items-center justify-center rounded-full shrink-0 size-[56px]">
                    <UserRoundSearch size={28} className="text-[#176547]" />
                  </div>
                  <p className="font-semibold text-[#23362f] text-[20px] text-center">Your result will appear here</p>
                  <p className="text-[#65736d] text-[15px] text-center">
                    Complete the customer details, then select Predict to see a call recommendation.
                  </p>
                </div>
              )}

              {!submitting && result && (
                <>
                  <div className="flex flex-col gap-[2px] w-full">
                    <p className="font-semibold text-[#176547] text-[64px] leading-[1.1]">
                      {(result.probability * 100).toFixed(0)}%
                    </p>
                    <p className="text-[#65736d] text-[15px]">Estimated chance of subscribing</p>
                  </div>
                  <div className="bg-[#eaf3ee] flex gap-[12px] items-center p-[16px] rounded-[6px] w-full">
                    <Phone size={20} className="text-[#176547]" />
                    <p className="flex-1 font-semibold text-[#176547] text-[22px]">{result.recommendation}</p>
                  </div>
                  <div className="flex items-center justify-between w-full">
                    <p className="text-[#65736d] text-[15px]">Predicted class</p>
                    <p className="font-semibold text-[#23362f] text-[16px]">
                      {result.predicted_class === "yes" ? "Yes" : "No"}
                    </p>
                  </div>
                  <div className="flex items-center justify-between w-full">
                    <p className="text-[#65736d] text-[15px]">Call priority</p>
                    <span className="flex gap-[6px] items-center px-[10px] py-[4px] rounded-full"
                      style={{ backgroundColor: PRIORITY_BADGE[result.priority].bg }}>
                      <span className="size-[5px] rounded-full" style={{ backgroundColor: PRIORITY_BADGE[result.priority].text }} />
                      <span className="font-semibold text-[13px]" style={{ color: PRIORITY_BADGE[result.priority].text }}>
                        {result.priority}
                      </span>
                    </span>
                  </div>
                  <p className="text-[#65736d] text-[14px]">
                    This result is {result.probability >= result.threshold ? "above" : "below"} the{" "}
                    {Math.round(result.threshold * 100)}% call threshold. A prediction is not a guarantee.
                  </p>
                </>
              )}
              <p className="text-[#65736d] text-[13px]">{result?.note ?? "Decision support only. Check contact preferences and opt-outs before calling."}</p>
            </div>

            <div className="bg-white border border-[#dde4df] rounded-[10px] p-[24px] flex flex-col gap-[18px] w-full">
              <p className="font-semibold text-[#23362f] text-[21px]">How priority works</p>
              <p className="text-[#65736d] text-[14px]">
                Based on the loaded model&apos;s call threshold · the same rules apply to individual customers and lists.
              </p>
              {high != null && medium != null ? (
                <>
                  <div className="flex items-center justify-between w-full">
                    <PriorityPill priority="High" />
                    <p className="text-[#65736d] text-[14px]">{high}% and above</p>
                  </div>
                  <div className="flex items-center justify-between w-full">
                    <PriorityPill priority="Medium" />
                    <p className="text-[#65736d] text-[14px]">{medium}% to below {high}%</p>
                  </div>
                  <div className="flex items-center justify-between w-full">
                    <PriorityPill priority="Low" />
                    <p className="text-[#65736d] text-[14px]">Below {medium}%</p>
                  </div>
                  <div className="text-[#65736d] text-[14px] w-full">
                    <p className="leading-[1.4]">{high}% or more → Yes · Call</p>
                    <p className="leading-[1.4]">Below {high}% → No · Do not call</p>
                  </div>
                </>
              ) : (
                <p className="text-[#65736d] text-[14px]">Available once a model is loaded.</p>
              )}
            </div>
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

function PriorityPill({ priority }: { priority: "High" | "Medium" | "Low" }) {
  const colors = PRIORITY_BADGE[priority];
  return (
    <span className="flex gap-[6px] items-center px-[10px] py-[4px] rounded-full" style={{ backgroundColor: colors.bg }}>
      <span className="size-[5px] rounded-full" style={{ backgroundColor: colors.text }} />
      <span className="font-semibold text-[13px]" style={{ color: colors.text }}>{priority}</span>
    </span>
  );
}
