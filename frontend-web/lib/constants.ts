// Dropdown options and field order, mirrored exactly from backend/schemas.py (ClientFeatures) so
// the form can only submit values the API will accept.

export const JOB_OPTIONS = ["admin.", "blue-collar", "entrepreneur", "housemaid", "management",
  "retired", "self-employed", "services", "student", "technician", "unemployed", "unknown"];
export const MARITAL_OPTIONS = ["divorced", "married", "single", "unknown"];
export const EDUCATION_OPTIONS = ["basic.4y", "basic.6y", "basic.9y", "high.school", "illiterate",
  "professional.course", "university.degree", "unknown"];
export const YES_NO_UNKNOWN_OPTIONS = ["yes", "no", "unknown"];
export const CONTACT_OPTIONS = ["cellular", "telephone"];
export const MONTH_OPTIONS = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"];
export const WEEKDAY_OPTIONS = ["mon", "tue", "wed", "thu", "fri"];
export const POUTCOME_OPTIONS = ["failure", "nonexistent", "success"];

export const REQUIRED_COLUMNS = ["age", "job", "marital", "education", "default", "housing", "loan",
  "contact", "month", "day_of_week", "campaign", "pdays", "previous", "poutcome", "emp_var_rate",
  "cons_price_idx", "cons_conf_idx", "euribor3m", "nr_employed"];

// Status-pill colors for priority, matching the Figma "How priority works" legend exactly
// (file VkyEQ4oNoz4UkvI8X9vfEE, node 2:10612).
export const PRIORITY_BADGE: Record<string, { bg: string; text: string }> = {
  High: { bg: "#eaf3ee", text: "#176547" },
  Medium: { bg: "#fbf3e2", text: "#865a12" },
  Low: { bg: "#faedec", text: "#9a4948" },
};

// Required CSV columns for batch upload, grouped to match the Figma "CSV column checklist"
// panel (file VkyEQ4oNoz4UkvI8X9vfEE, node 2:10839).
export const COLUMN_GROUPS: { title: string; columns: string[] }[] = [
  { title: "Client profile", columns: ["age", "job", "marital", "education"] },
  { title: "Credit / loans", columns: ["default", "housing", "loan"] },
  { title: "Contact", columns: ["contact", "month", "day_of_week"] },
  { title: "Campaign history", columns: ["campaign", "pdays", "previous", "poutcome"] },
  { title: "Economic context", columns: ["emp_var_rate", "cons_price_idx", "cons_conf_idx", "euribor3m", "nr_employed"] },
];

// Numeric bounds enforced by backend/schemas.py ClientFeatures, shared by the Single Prediction
// form's client-side validation and the Batch Upload page's row-error explanations.
export const RANGE_RULES: { key: string; label: string; min: number; max: number }[] = [
  { key: "age", label: "Age", min: 18, max: 100 },
  { key: "campaign", label: "Contacts this campaign", min: 1, max: 60 },
  { key: "pdays", label: "Days since prior contact", min: 0, max: 999 },
  { key: "previous", label: "Prior campaign contacts", min: 0, max: 20 },
];

const CATEGORICAL_FIELDS: Record<string, string[]> = {
  job: JOB_OPTIONS, marital: MARITAL_OPTIONS, education: EDUCATION_OPTIONS,
  default: YES_NO_UNKNOWN_OPTIONS, housing: YES_NO_UNKNOWN_OPTIONS, loan: YES_NO_UNKNOWN_OPTIONS,
  contact: CONTACT_OPTIONS, month: MONTH_OPTIONS, day_of_week: WEEKDAY_OPTIONS, poutcome: POUTCOME_OPTIONS,
};

// Same example values used as the Single Prediction form's economic-context defaults.
const ECONOMIC_FIELD_EXAMPLES: Record<string, number> = {
  emp_var_rate: -1.8, cons_price_idx: 92.893, cons_conf_idx: -46.2, euribor3m: 1.313, nr_employed: 5099.1,
};

// Translates a field name into plain-language guidance, for the Batch Upload "Rows to correct"
// table. Covers both out-of-range and wrong-type values with the same actionable message, since
// the fix is the same either way: enter a valid value of the right kind for that field.
export function explainFieldError(field: string): string {
  const range = RANGE_RULES.find((r) => r.key === field);
  if (range) {
    const noun = field === "age" ? "an age" : "a value";
    return `Enter ${noun} between ${range.min} and ${range.max}.`;
  }
  const options = CATEGORICAL_FIELDS[field];
  if (options) {
    return options.length <= 2
      ? `Use ${options.join(" or ")}.`
      : `Use ${options.slice(0, -1).join(", ")}, or ${options[options.length - 1]}.`;
  }
  const example = ECONOMIC_FIELD_EXAMPLES[field];
  if (example != null) return `Enter a numeric value, e.g. ${example}.`;
  return "Check this value and try again.";
}

// Design tokens pulled from the real Figma design (file VkyEQ4oNoz4UkvI8X9vfEE,
// frame "Home - Dashboard", node 2:10479) via get_design_context.
export const COLORS = {
  textPrimary: "#23362f",
  textSecondary: "#65736d",
  accent: "#176547",
  accentBg: "#eaf3ee",
  badgeBg: "#eef1ef",
  border: "#dde4df",
  surface: "#ffffff",
  pageBg: "#f5f7f6",
};
