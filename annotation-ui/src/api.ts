export type AnnotationStatus = "accepted" | "modified" | "rejected";

export type Assignment = {
  assignment_id: number;
  annotator_id: string;
  item: {
    question_id: string;
    verdict_id: string;
    question: string;
    gold_answer: string;
    gold_paragraphs: string[];
    question_type: string;
    source_status?: string | null;
  };
};

export type AdminSummary = {
  total_items: number;
  total_assignments: number;
  completed_assignments: number;
  completed_items: number;
  cohen_kappa: number | null;
  annotators: Array<{
    annotator_id: string;
    display_name: string;
    assigned: number;
    completed: number;
  }>;
  status_counts: Record<string, number>;
  agreement_items: Array<{
    question_id: string;
    verdict_id: string;
    votes: Record<string, number>;
    consensus_status: AnnotationStatus | null;
    is_tie: boolean;
  }>;
};

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://127.0.0.1:8000";

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!response.ok) {
    const body = await response.text();
    throw new Error(body || `Request failed: ${response.status}`);
  }
  return response.json() as Promise<T>;
}

export const api = {
  listAnnotators: () => request<Array<{ annotator_id: string; display_name: string }>>("/api/annotators"),
  bootstrap: (payload: {
    dataset_path: string;
    annotator_ids: string[];
    assignments_per_item: number;
    seed: number;
    reset_existing: boolean;
  }) => request("/api/admin/bootstrap", { method: "POST", body: JSON.stringify(payload) }),
  summary: () => request<AdminSummary>("/api/admin/summary"),
  nextTask: (annotatorId: string) => request<{ assignment: Assignment | null }>(`/api/tasks/next/${annotatorId}`),
  submitAnnotation: (payload: {
    assignment_id: number;
    annotator_id: string;
    status: AnnotationStatus;
    edited_question?: string | null;
    edited_gold_answer?: string | null;
    edited_gold_paragraphs?: string[] | null;
    notes?: string | null;
  }) => request<{ ok: boolean; next_assignment: Assignment | null }>("/api/annotations", {
    method: "POST",
    body: JSON.stringify(payload),
  }),
};
