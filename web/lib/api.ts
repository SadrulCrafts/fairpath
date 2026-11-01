import type { Meta } from "./types";

const BASE = (process.env.NEXT_PUBLIC_API_BASE ?? "").replace(/\/$/, "");

export class ApiError extends Error {
  constructor(message: string, public status: number, public details?: unknown) { super(message); }
}

async function handle<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let body: { detail?: unknown; errors?: unknown } = {};
    try { body = await res.json(); } catch { /* not json */ }
    const detail = typeof body.detail === "string" ? body.detail : `Request failed (${res.status})`;
    throw new ApiError(detail, res.status, body.errors);
  }
  return res.json() as Promise<T>;
}

const cache = new Map<string, Promise<unknown>>();

/** GET with an in-memory cache: reports are static for a given deployment. */
export function getJSON<T>(path: string, { fresh = false } = {}): Promise<T> {
  if (!fresh && cache.has(path)) return cache.get(path) as Promise<T>;
  const p = fetch(`${BASE}${path}`).then((r) => handle<T>(r));
  cache.set(path, p);
  p.catch(() => cache.delete(path));
  return p;
}

export function postJSON<T>(path: string, body: unknown, signal?: AbortSignal): Promise<T> {
  return fetch(`${BASE}${path}`, {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body), signal,
  }).then((r) => handle<T>(r));
}

export async function postFile(path: string, file: File): Promise<Response> {
  const fd = new FormData();
  fd.append("file", file);
  return fetch(`${BASE}${path}`, { method: "POST", body: fd });
}

export const api = {
  meta: () => getJSON<Meta>("/api/meta"),
  url: (path: string) => `${BASE}${path}`,
};
