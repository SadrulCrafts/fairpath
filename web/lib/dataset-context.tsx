"use client";
import * as React from "react";
import { api } from "./api";
import type { DatasetMeta, Meta } from "./types";

interface Ctx { meta: Meta | null; error: string | null; dataset: string; setDataset: (d: string) => void; current: DatasetMeta | null }
const DatasetContext = React.createContext<Ctx | null>(null);
const KEY = "fairpath.dataset";

export function DatasetProvider({ children }: { children: React.ReactNode }) {
  const [meta, setMeta] = React.useState<Meta | null>(null);
  const [error, setError] = React.useState<string | null>(null);
  const [dataset, setDatasetState] = React.useState("dropout");

  React.useEffect(() => {
    try { const saved = window.localStorage.getItem(KEY); if (saved) setDatasetState(saved); } catch { /* storage blocked */ }
    api.meta().then(setMeta).catch((e: Error) => setError(e.message));
  }, []);

  const setDataset = React.useCallback((d: string) => {
    setDatasetState(d);
    try { window.localStorage.setItem(KEY, d); } catch { /* storage blocked */ }
  }, []);

  const current = meta?.datasets.find((d) => d.name === dataset) ?? meta?.datasets[0] ?? null;
  return (
    <DatasetContext.Provider value={{ meta, error, dataset: current?.name ?? dataset, setDataset, current }}>
      {children}
    </DatasetContext.Provider>
  );
}

export function useDataset() {
  const ctx = React.useContext(DatasetContext);
  if (!ctx) throw new Error("useDataset outside provider");
  return ctx;
}

/** Fetch something that depends on the selected dataset, with loading and error state. */
export function useData<T>(loader: (ds: string) => Promise<T>, deps: unknown[] = []) {
  const { dataset, meta } = useDataset();
  const [state, setState] = React.useState<{ data: T | null; error: string | null; loading: boolean }>({ data: null, error: null, loading: true });
  React.useEffect(() => {
    if (!meta) return;
    let alive = true;
    setState((s) => ({ ...s, loading: true, error: null }));
    loader(dataset)
      .then((data) => alive && setState({ data, error: null, loading: false }))
      .catch((e: Error) => alive && setState({ data: null, error: e.message, loading: false }));
    return () => { alive = false; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dataset, meta, ...deps]);
  return state;
}
