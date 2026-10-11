// URL query parameters. All positions are in the pipeline's scene frame (metres, z up, +x east, +y north).
//
//   ?scan=<id>              which scan (data/scans.json, written by `scanmodel export`): its folder data/scans/<id>/
//                           has its own splats.json, model.glb and views.json. Default: scans.json's default.
//                           Without scans.json: data/ itself. Scans have their own scene frames
//   ?view=<name>            start at a named view of the scan (views.json, from its scene/views.json)
//   ?splat=<id>|<url>|none  splat to show first: an id from the scan's splats.json, any splat URL (.spz / .ply
//                           / .splat / .ksplat / .sog), or none. Default: splats.json's default
//   ?glb=<url>|none         glTF structure (the Blender model); default the scan's model.glb
//   ?model=0|1              show the model at start; default: only when no splat is shown
//   ?splatUp=z|y            up axis of the splat file (default z: trained in the scene frame)
//   ?glbUp=y|z              up axis of the glTF file (default y: glTF's own convention; exporters convert z-up to y-up)
//   ?cam=x,y,z              initial camera position (scene frame)
//   ?look=x,y,z             initial look-at point (scene frame)
//   ?pose=m00,…,m33         exact camera: 16 numbers, row-major camera-to-world with OpenGL camera axes in the
//                           scene frame (a nerfstudio `transform_matrix`). Freezes the camera (for scripted renders)
//   ?fov=deg                vertical field of view (default 70)
//
// C copies the current view as ?pose=…&fov=… (exact, frozen); shift+C as ?cam=…&look=…&fov=… (walkable).
// `scanmodel view add NAME "<url>"` takes either.

export type Up = "y" | "z";

export interface Params {
  scan?: string; // id from data/scans.json; undefined: its default
  view?: string; // a named view of the scan
  splat?: string; // id, URL or "none"; undefined: the catalogue's default
  glb?: string[]; // candidates, first one that exists wins; undefined: the scan's model.glb
  model?: boolean;
  splatUp: Up;
  glbUp: Up;
  cam?: [number, number, number];
  look?: [number, number, number];
  pose?: number[]; // 16, row-major
  fov?: number;
  explicitGlb: boolean;
}

export const SCANS_URL = "data/scans.json";

function vec3(text: string | null): [number, number, number] | undefined {
  if (!text) return undefined;
  const v = text.split(",").map(Number);
  return v.length === 3 && v.every(Number.isFinite) ? [v[0], v[1], v[2]] : undefined;
}

function matrix(text: string | null): number[] | undefined {
  const v = text?.split(",").map(Number);
  return v?.length === 16 && v.every(Number.isFinite) ? v : undefined;
}

function positive(text: string | null): number | undefined {
  const v = Number(text ?? NaN);
  return Number.isFinite(v) && v > 0 ? v : undefined;
}

function up(text: string | null, fallback: Up): Up {
  return text === "y" || text === "z" ? text : fallback;
}

function sources(value: string | null): string[] | undefined {
  if (value === null) return undefined;
  return value === "" || value === "none" ? [] : [value];
}

function flag(text: string | null): boolean | undefined {
  if (text === null) return undefined;
  return !["0", "false", "off", "no"].includes(text.toLowerCase());
}

export function readParams(search = window.location.search): Params {
  const q = new URLSearchParams(search);
  const splat = q.get("splat");
  return {
    scan: q.get("scan") || undefined,
    view: q.get("view") || undefined,
    splat: splat === null ? undefined : splat === "" ? "none" : splat,
    glb: sources(q.get("glb")),
    model: flag(q.get("model")),
    splatUp: up(q.get("splatUp"), "z"),
    glbUp: up(q.get("glbUp"), "y"),
    cam: vec3(q.get("cam")),
    look: vec3(q.get("look")),
    pose: matrix(q.get("pose")),
    fov: positive(q.get("fov")),
    explicitGlb: q.has("glb"),
  };
}

/** Shortest decimal for `x` rounded to `digits` places, without -0 (keeps copied URLs short and stable). */
const num = (x: number, digits = 6) => String(Number(x.toFixed(digits)) + 0);

/** What a copied view URL carries; position fields are in the scene frame, as readParams expects them. */
export interface View {
  pose?: number[]; // 16, row-major camera-to-world, OpenGL camera axes (exact; freezes the camera on load)
  cam?: [number, number, number]; // or a walkable camera: position…
  look?: [number, number, number]; // …and a point it looks at
  fov: number;
  splat?: string; // id or URL of the shown splat, "none"
  model?: boolean;
  scan?: string; // the scan shown (its scene frame is what the positions mean)
}

/** `href` with the view's parameters set (the inverse of readParams); other parameters are kept. */
export function viewHref(href: string, view: View): string {
  const url = new URL(href);
  const q = url.searchParams;
  for (const key of ["pose", "cam", "look", "view"]) q.delete(key);
  if (view.scan !== undefined) q.set("scan", view.scan);
  if (view.pose) q.set("pose", view.pose.map((x) => num(x)).join(","));
  if (view.cam) q.set("cam", view.cam.map((x) => num(x)).join(","));
  if (view.look) q.set("look", view.look.map((x) => num(x)).join(","));
  q.set("fov", num(view.fov, 3));
  if (view.splat !== undefined) q.set("splat", view.splat);
  if (view.model !== undefined) q.set("model", view.model ? "1" : "0");
  url.search = q.toString().replace(/%2C/gi, ","); // commas stay readable; both parsers accept them
  return url.toString();
}

/** True if the server has the file (Vite's dev server answers unknown paths with index.html). */
async function exists(url: string): Promise<boolean> {
  try {
    const res = await fetch(url, { method: "HEAD" });
    return res.ok && !(res.headers.get("content-type") ?? "").includes("text/html");
  } catch {
    return false; // network error
  }
}

/** First URL that the server actually has. */
export async function firstExisting(urls: string[]): Promise<string | undefined> {
  for (const url of urls) if (await exists(url)) return url;
  return undefined;
}

/** One trained splat, as `scanmodel export` describes it in data/splats.json. */
export interface SplatEntry {
  id: string;
  label: string;
  url: string;
  trainer?: string;
  params?: Record<string, unknown>;
  summary?: Record<string, unknown>;
  trained_at?: string | null;
  file_at?: string;
  size_mb?: number;
  current?: boolean; // trained on the current dataset
}

export interface Catalog {
  default?: string | null;
  splats: SplatEntry[];
}

/** JSON at `url`, or undefined (missing: Vite's dev server answers unknown paths with index.html). */
async function fetchJson<T>(url: string): Promise<T | undefined> {
  try {
    const res = await fetch(url);
    if (res.ok && (res.headers.get("content-type") ?? "").includes("json")) return (await res.json()) as T;
  } catch {
    // network error
  }
  return undefined;
}

/** One exported scan, as `scanmodel export` lists it in data/scans.json. */
export interface ScanEntry {
  id: string;
  label?: string;
  base: string; // its folder as the page fetches it: data/scans/<id>/
  default_splat?: string | null;
  splats?: number;
  views?: number;
  exported_at?: string;
}

/** The scan this page shows, and the others it can switch to. */
export interface ScanChoice {
  scans: ScanEntry[]; // empty: an older export without scans.json
  current?: ScanEntry;
  base: string; // where splats.json, model.glb and views.json are
  error?: string; // ?scan= named a scan that isn't there
}

/** ?scan= (else scans.json's default) from data/scans.json; without it, data/ as an empty export. */
export async function loadScans(want: string | undefined): Promise<ScanChoice> {
  const index = await fetchJson<{ default?: string; scans?: ScanEntry[] }>(SCANS_URL);
  const scans = Array.isArray(index?.scans) ? index.scans : [];
  if (!scans.length) return { scans, base: "data/", error: want ? `no scans.json: ?scan=${want} ignored` : undefined };
  const byId = (id: string | undefined) => scans.find((s) => s.id === id);
  const current = byId(want) ?? byId(index?.default) ?? scans[0];
  const error = want && !byId(want) ? `no scan ${want} (have ${scans.map((s) => s.id).join(", ")})` : undefined;
  return { scans, current, base: current.base, error };
}

/** The scan's splats.json; none if missing. */
export async function loadCatalog(base: string): Promise<Catalog> {
  const cat = await fetchJson<Catalog>(`${base}splats.json`);
  return cat && Array.isArray(cat.splats) ? cat : { splats: [] };
}

/** A named view: `pose` as `?pose=` takes it (row-major camera-to-world, OpenGL axes, scene frame). */
export interface NamedView {
  name: string;
  note?: string;
  pose: number[];
  fov: number;
}

/** The scan's views.json (`scanmodel export`, from its scene/views.json); none if missing. */
export async function loadViews(base: string): Promise<NamedView[]> {
  const v = await fetchJson<{ views?: NamedView[] }>(`${base}views.json`);
  return (v?.views ?? []).filter((x) => Array.isArray(x.pose) && x.pose.length === 16);
}

/** This page's URL for another scan: only the parameters that mean the same in every scan's frame stay. */
export function scanHref(href: string, scan: string): string {
  const url = new URL(href);
  for (const key of ["pose", "cam", "look", "splat", "view", "glb"]) url.searchParams.delete(key);
  url.searchParams.set("scan", scan);
  return url.toString();
}
