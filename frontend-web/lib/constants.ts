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
