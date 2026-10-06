// Client for the FastAPI backend (backend/app.py). Mirrors the request/response shapes in
// backend/schemas.py exactly, so this is the single source of truth for the Next.js frontend.

export const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8001";

export type ClientFeatures = {
  age: number;
  job: string;
  marital: string;
  education: string;
  default: string;
  housing: string;
  loan: string;
  contact: string;
  month: string;
  day_of_week: string;
  campaign: number;
  pdays: number;
  previous: number;
  poutcome: string;
  emp_var_rate: number;
  cons_price_idx: number;
  cons_conf_idx: number;
  euribor3m: number;
  nr_employed: number;
};

export type PredictionResponse = {
  probability: number;
  predicted_class: "yes" | "no";
  recommendation: "Call" | "Do not call";
  priority: "High" | "Medium" | "Low";
  threshold: number;
  model_name: string;
  note: string;
};

export type BatchResultRow = {
  rank: number;
  probability: number;
  predicted_class: "yes" | "no";
  recommendation: "Call" | "Do not call";
  priority: "High" | "Medium" | "Low";
  [extra: string]: unknown; // e.g. client_id, passed through unchanged by the API
};

export type BatchPredictionResponse = {
  count: number;
  threshold: number;
  model_name: string;
  results: BatchResultRow[];
  note: string;
};

export type HealthResponse = { status: string; model_loaded: boolean };
export type ModelInfoResponse = { model_name: string; threshold: number; input_columns: string[] };

export class ApiError extends Error {
  detail: unknown;
  status: number;
  constructor(status: number, detail: unknown) {
    super(typeof detail === "string" ? detail : JSON.stringify(detail));
    this.status = status;
    this.detail = detail;
  }
}

async function parseErrorDetail(response: Response): Promise<unknown> {
  try {
    const body = await response.json();
    return body.detail ?? body;
  } catch {
    return response.statusText;
  }
}

export async function fetchHealth(): Promise<HealthResponse> {
  const response = await fetch(`${API_URL}/health`, { cache: "no-store" });
  if (!response.ok) throw new ApiError(response.status, await parseErrorDetail(response));
  return response.json();
}

export async function fetchModelInfo(): Promise<ModelInfoResponse> {
  const response = await fetch(`${API_URL}/model-info`, { cache: "no-store" });
  if (!response.ok) throw new ApiError(response.status, await parseErrorDetail(response));
  return response.json();
}

export async function predict(client: ClientFeatures): Promise<PredictionResponse> {
  const response = await fetch(`${API_URL}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(client),
  });
  if (!response.ok) throw new ApiError(response.status, await parseErrorDetail(response));
  return response.json();
}

export async function predictBatch(file: File): Promise<BatchPredictionResponse> {
  const form = new FormData();
  form.append("file", file);
  const response = await fetch(`${API_URL}/predict-batch`, { method: "POST", body: form });
  if (!response.ok) throw new ApiError(response.status, await parseErrorDetail(response));
  return response.json();
}

export function batchTemplateUrl(): string {
  return `${API_URL}/predict-batch/template`;
}
