import type { NamedView, ScanEntry, SplatEntry } from "./params";

/** The layer panel (top right): which scan and named view, which splat, its training parameters, and the
 * Blender model overlay. */
export interface PanelHandlers {
  selectScan(id: string): void; // reloads the page on that scan
  selectView(index: number): void;
  selectSplat(index: number): void; // -1: none
  setModel(on: boolean): void;
  setOpacity(opacity: number): void;
  setWireframe(on: boolean): void;
  copyView(walkable: boolean): void; // a URL with the current camera: exact ?pose=, or walkable ?cam=&look=
}

const el = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

const num = (v: unknown) => (typeof v === "number" && Number.isFinite(v) ? v : undefined);

function when(iso: string | null | undefined): string | undefined {
  if (!iso) return undefined;
  const d = new Date(iso);
  return Number.isNaN(d.getTime())
    ? undefined
    : d.toLocaleString(undefined, { year: "numeric", month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
}

function escape(text: string): string {
  return text.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c] ?? c);
}

/** What the panel says about a splat: trainer, steps, resolution, size, metrics, date. */
export function describe(entry: SplatEntry, loaded?: number): string {
  const s = entry.summary ?? {};
  const p = entry.params ?? {};
  const rows: [string, string][] = [];
  const trainer = entry.trainer ?? (typeof s.trainer === "string" ? s.trainer : undefined);
  if (trainer) rows.push(["trainer", trainer]);
  const iters = num(s.iterations) ?? num(p.iterations);
  if (iters) rows.push(["steps", iters.toLocaleString()]);
  const down = num(s.downscale) ?? num(p.downscale);
  rows.push(["resolution", down && down > 1 ? `1/${down}` : "full (dataset max side)"]);
  const splats = loaded ?? num(s.splats);
  if (splats) rows.push(["splats", splats.toLocaleString()]);
  if (entry.size_mb) rows.push(["file", `${entry.size_mb.toLocaleString()} MB`]);
  const psnr = num(s.psnr);
  const ssim = num(s.ssim);
  if (psnr || ssim) {
    const every = num(p.eval_every);
    const held = every ? ` (every ${every}th view held out)` : "";
    rows.push(["PSNR / SSIM", `${psnr?.toFixed(2) ?? "–"} / ${ssim?.toFixed(3) ?? "–"}${held}`]);
  }
  const wall = num(s.wall_s);
  const date = when(entry.trained_at) ?? when(entry.file_at);
  if (date) rows.push(["trained", wall ? `${date}, ${(wall / 60).toFixed(1)} min` : date]);
  const extra = Array.isArray(p.extra) && p.extra.length ? p.extra.join(" ") : "";
  if (extra) rows.push(["extra args", extra]);
  for (const [k, v] of Object.entries(s)) {
    // anything else a variant's sidecar json adds (e.g. a cleaning step's counts)
    if (["trainer", "iterations", "downscale", "splats", "ply_mb", "psnr", "ssim", "wall_s"].includes(k)) continue;
    if (typeof v === "number" || typeof v === "string" || typeof v === "boolean") rows.push([k, String(v)]);
  }
  let html = `<table>${rows.map(([k, v]) => `<tr><td>${escape(k)}</td><td>${escape(v)}</td></tr>`).join("")}</table>`;
  if (entry.current === false) html += `<div class="warn">trained on an older dataset (re-run scanmodel splat --train)</div>`;
  return html;
}

export class Panel {
  private readonly select = el<HTMLSelectElement>("splat-select");
  private readonly info = el<HTMLDivElement>("splat-info");
  private readonly model = el<HTMLInputElement>("model-on");
  private readonly opacity = el<HTMLInputElement>("model-opacity");
  private readonly opacityValue = el<HTMLSpanElement>("model-opacity-value");
  private readonly wire = el<HTMLInputElement>("model-wire");
  private readonly root = el<HTMLDivElement>("panel");
  private readonly view = el<HTMLSelectElement>("view-select");

  constructor(
    private readonly entries: SplatEntry[],
    handlers: PanelHandlers,
    hasModel: boolean,
    scans: ScanEntry[] = [],
    currentScan?: string,
    views: NamedView[] = [],
  ) {
    // scan: shown once scans.json lists one; a choice once it lists two
    const scanSelect = el<HTMLSelectElement>("scan-select");
    el<HTMLDivElement>("scan-row").hidden = scans.length === 0;
    scanSelect.innerHTML = scans
      .map((s) => `<option value="${escape(s.id)}" title="${escape(s.label ?? s.id)}">${escape(s.label ?? s.id)}</option>`)
      .join("");
    if (currentScan) scanSelect.value = currentScan;
    scanSelect.disabled = scans.length < 2;
    scanSelect.addEventListener("change", () => {
      scanSelect.blur();
      handlers.selectScan(scanSelect.value);
    });
    el<HTMLDivElement>("view-row").hidden = views.length === 0;
    this.view.innerHTML = [`<option value="-1">–</option>`]
      .concat(views.map((v, i) => `<option value="${i}" title="${escape(v.note ?? "")}">${escape(v.name)}</option>`))
      .join("");
    this.view.addEventListener("change", () => {
      this.view.blur();
      if (Number(this.view.value) >= 0) handlers.selectView(Number(this.view.value));
    });
    const options = [`<option value="-1">none</option>`];
    entries.forEach((e, i) => {
      const key = i < 9 ? ` [${i + 1}]` : "";
      const stale = e.current === false ? " (older dataset)" : "";
      options.push(`<option value="${i}">${escape(e.label)}${stale}${key}</option>`);
    });
    this.select.innerHTML = options.join("");
    this.select.addEventListener("change", () => {
      handlers.selectSplat(Number(this.select.value));
      this.select.blur(); // arrow keys walk again instead of changing the selection
    });
    for (const input of [this.model, this.opacity, this.wire]) input.disabled = !hasModel;
    if (!hasModel) el<HTMLDivElement>("model-section").title = "no model.glb";
    this.model.addEventListener("change", () => handlers.setModel(this.model.checked));
    this.opacity.addEventListener("input", () => {
      this.opacityValue.textContent = `${Math.round(Number(this.opacity.value) * 100)}%`;
      handlers.setOpacity(Number(this.opacity.value));
    });
    this.wire.addEventListener("change", () => handlers.setWireframe(this.wire.checked));
    el<HTMLButtonElement>("panel-toggle").addEventListener("click", () => this.toggle());
    // done with a control: give the keys back to walking and the shortcuts
    for (const c of [this.model, this.opacity, this.wire]) c.addEventListener("change", () => c.blur());
    el<HTMLButtonElement>("panel-toggle").addEventListener("click", (e) => (e.currentTarget as HTMLElement).blur());
    el<HTMLButtonElement>("copy-view").addEventListener("click", (e) => {
      (e.currentTarget as HTMLElement).blur();
      handlers.copyView(e.shiftKey);
    });
  }

  toggle(): void {
    const collapsed = this.root.classList.toggle("collapsed");
    el<HTMLButtonElement>("panel-toggle").textContent = collapsed ? "+" : "–";
  }

  showSplat(index: number, state: "loading" | "loaded" | "error", detail?: string, loaded?: number): void {
    this.select.value = String(index);
    if (index < 0) {
      this.info.innerHTML = this.entries.length ? "" : `<div class="dim">no splat: run scanmodel splat --train, then scanmodel export</div>`;
      return;
    }
    const entry = this.entries[index];
    const head =
      state === "loading" ? `<div class="dim">loading…</div>` : state === "error" ? `<div class="err">${escape(detail ?? "load failed")}</div>` : "";
    this.info.innerHTML = head + describe(entry, loaded);
  }

  showView(index: number): void {
    this.view.value = String(index);
  }

  showModel(on: boolean, opacity: number, wireframe: boolean): void {
    this.model.checked = on;
    this.opacity.value = String(opacity);
    this.opacityValue.textContent = `${Math.round(opacity * 100)}%`;
    this.wire.checked = wireframe;
  }
}
