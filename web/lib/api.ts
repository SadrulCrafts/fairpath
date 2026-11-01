import * as React from "react";
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

/**
 * Read-only data (metadata, schemas, audit reports) is also exported as static JSON with the dashboard
 * (scripts/export_static_data.py). Reading that first means the audit pages open instantly even when the
 * API is asleep on a free host; the API is the fallback.
 */
function staticPath(path: string): string | null {
  if (path === "/api/meta") return "/data/meta.json";
  if (path === "/api/reports/overview") return "/data/overview.json";
  if (path === "/api/reports/thesis") return "/data/thesis.json";
  let m = path.match(/^\/api\/reports\/([a-z_]+)\/([a-z_]+)$/);
  if (m) return `/data/${m[1]}/${m[2]}.json`;
  m = path.match(/^\/api\/datasets\/([a-z_]+)\/(schema|examples)$/);
  if (m) return `/data/${m[1]}/${m[2]}.json`;
  return null;
}

/** GET with an in-memory cache: reports are static for a given deployment. */
export function getJSON<T>(path: string, { fresh = false } = {}): Promise<T> {
  if (!fresh && cache.has(path)) return cache.get(path) as Promise<T>;
  const fromApi = () => fetch(`${BASE}${path}`).then((r) => handle<T>(r));
  const sp = staticPath(path);
  const p = sp
    ? fetch(sp).then((r) => (r.ok ? (r.json() as Promise<T>) : fromApi())).catch(() => fromApi())
    : fromApi();
  cache.set(path, p);
  p.catch(() => cache.delete(path));
  return p;
}

/** True after `ms` while a promise is still pending: used to explain cold starts on free hosting. */
export function useSlow(pending: boolean, ms = 4000) {
  const [slow, setSlow] = React.useState(false);
  React.useEffect(() => {
    if (!pending) { setSlow(false); return; }
    const t = setTimeout(() => setSlow(true), ms);
    return () => clearTimeout(t);
  }, [pending, ms]);
  return slow;
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
