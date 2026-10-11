import { SparkRenderer, SplatMesh } from "@sparkjsdev/spark";
import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { Panel } from "./panel";
import {
  firstExisting,
  loadCatalog,
  loadScans,
  loadViews,
  readParams,
  scanHref,
  viewHref,
  type NamedView,
  type ScanEntry,
  type SplatEntry,
  type Up,
} from "./params";
import { WalkControls } from "./walk";

type V3 = [number, number, number];

// three.js is y-up; the pipeline's scene frame is z-up (+x east, +y north). Anything authored z-up is
// rotated -90 degrees about x: scene (x, y, z) -> three (x, z, -y). East stays +x, north is -z, up is +y.
// This is the same mapping glTF exporters apply, so a glTF exported from the .blend lines up with the splat.
const Z_UP_TO_Y_UP = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), -Math.PI / 2);
const sceneToThree = ([x, y, z]: V3) => new THREE.Vector3(x, z, -y);
const threeToScene = (v: THREE.Vector3): V3 => [v.x, -v.z, v.y];

const statusEl = document.getElementById("status")!;
const poseEl = document.getElementById("pose")!;
const statusLines = new Map<string, string>(); // one line per topic, so switching splats doesn't pile up lines
function status(key: string, msg: string, error = false): void {
  if (error) console.error(msg);
  statusLines.set(key, error ? `<span class="err">${msg}</span>` : msg);
  statusEl.innerHTML = [...statusLines.values()].join("<br>");
}

// --- renderer, scene, camera --------------------------------------------------------------------
const renderer = new THREE.WebGLRenderer({ antialias: false }); // Spark recommends no MSAA
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x202024);
scene.add(new SparkRenderer({ renderer }));
scene.add(new THREE.HemisphereLight(0xffffff, 0x404040, 1.5));
const sun = new THREE.DirectionalLight(0xffffff, 1.5);
sun.position.set(3, 10, 4);
scene.add(sun);

const camera = new THREE.PerspectiveCamera(70, window.innerWidth / window.innerHeight, 0.01, 1000);
let frozen = false; // ?pose: the camera stays where the URL put it

/** Exact camera from a scene-frame camera-to-world (row-major, OpenGL axes as three.js cameras use). Scripts
 * call it as `window.setPose(m)` to render many views without reloading the splat. */
function setPose(m16: number[]): void {
  const m = new THREE.Matrix4().set(...(m16 as Parameters<THREE.Matrix4["set"]>));
  m.premultiply(new THREE.Matrix4().makeRotationFromQuaternion(Z_UP_TO_Y_UP));
  m.decompose(camera.position, camera.quaternion, new THREE.Vector3());
  frozen = true;
}
(window as unknown as { setPose: typeof setPose }).setPose = setPose;

/** The current camera as setPose takes it (its inverse): scene-frame camera-to-world, row-major, OpenGL axes. */
function getPose(): number[] {
  camera.updateMatrixWorld();
  const m = camera.matrixWorld.clone().premultiply(new THREE.Matrix4().makeRotationFromQuaternion(Z_UP_TO_Y_UP).invert());
  const rows = m.transpose().elements.slice(); // three.js stores column-major: transposed, it reads row by row
  rows.splice(12, 4, 0, 0, 0, 1);
  return rows;
}
const walk = new WalkControls(camera, renderer.domElement);

// Floor grid (1 m) and scene-frame axes: red = east (+x), green = north (+y), blue = up (+z).
const helpers = new THREE.Group();
helpers.add(new THREE.GridHelper(20, 20, 0x666666, 0x333333));
const axes = new THREE.AxesHelper(1);
axes.quaternion.copy(Z_UP_TO_Y_UP);
helpers.add(axes);
scene.add(helpers);

const content = new THREE.Group(); // everything loaded; used for framing
scene.add(content);

function orient(object: THREE.Object3D, up: Up): THREE.Object3D {
  if (up === "z") object.quaternion.copy(Z_UP_TO_Y_UP);
  return object;
}

// --- layers: one splat at a time, the Blender model as an optional overlay -------------------------
const params = readParams();
if (params.fov) {
  camera.fov = params.fov;
  camera.updateProjectionMatrix();
}
let splatMesh: SplatMesh | null = null;
let splatIndex = -1; // selected entry, -1: none
let splatToken = 0; // a newer selection wins over a load still in flight
let model: THREE.Group | null = null;
let modelOpacity = 1;
let modelWireframe = false;
let entries: SplatEntry[] = [];
let panel: Panel | null = null;
let scan: ScanEntry | undefined; // undefined: an older export without scans.json
let views: NamedView[] = [];

async function showSplat(index: number): Promise<void> {
  const token = ++splatToken;
  splatIndex = entries[index] ? index : -1;
  if (splatMesh) {
    content.remove(splatMesh);
    splatMesh.dispose();
    splatMesh = null;
  }
  const entry = entries[index];
  if (!entry) {
    panel?.showSplat(-1, "loaded");
    status("splat", "splat: none");
    return;
  }
  panel?.showSplat(index, "loading");
  status("splat", `splat: loading ${entry.label}…`);
  const mesh = new SplatMesh({ url: entry.url });
  try {
    await mesh.initialized;
  } catch (e: unknown) {
    mesh.dispose();
    if (token === splatToken) {
      panel?.showSplat(index, "error", String(e));
      status("splat", `splat ${entry.url}: ${String(e)}`, true);
    }
    return;
  }
  if (token !== splatToken) {
    mesh.dispose();
    return;
  }
  splatMesh = mesh;
  content.add(orient(mesh, params.splatUp));
  panel?.showSplat(index, "loaded", undefined, mesh.numSplats);
  status("splat", `splat: ${entry.label} (${mesh.numSplats.toLocaleString()} splats)`);
}

async function loadModel(url: string, up: Up): Promise<void> {
  const gltf = await new GLTFLoader().loadAsync(url);
  const root = new THREE.Group();
  root.add(gltf.scene);
  gltf.scene.traverse((o) => {
    // own material per mesh, so opacity and wireframe can change without touching shared ones
    if (o instanceof THREE.Mesh) o.material = Array.isArray(o.material) ? o.material.map((m) => m.clone()) : o.material.clone();
  });
  content.add(orient(root, up));
  model = root;
  status("glb", `model: ${url}`);
}

function styleModel(): void {
  if (!model) return;
  const see = modelOpacity < 1;
  model.traverse((o) => {
    if (!(o instanceof THREE.Mesh)) return;
    // Spark draws the splats as one transparent pass without depth writes: a see-through model has
    // to come after it, or the splats paint over it
    o.renderOrder = see ? 10 : 0;
    for (const m of Array.isArray(o.material) ? o.material : [o.material]) {
      const mat = m as THREE.Material & { wireframe?: boolean };
      mat.transparent = see;
      mat.opacity = modelOpacity;
      mat.depthWrite = !see; // see-through: the splat behind still shows
      if ("wireframe" in mat) mat.wireframe = modelWireframe;
      mat.needsUpdate = true;
    }
  });
}

function setModel(on: boolean): void {
  if (model) model.visible = on;
  panel?.showModel(on, modelOpacity, modelWireframe);
}

function frameAll(): void {
  content.updateMatrixWorld(true);
  const box = new THREE.Box3();
  for (const child of content.children) {
    if (child instanceof SplatMesh) {
      box.union(child.getBoundingBox().applyMatrix4(child.matrixWorld));
    } else {
      box.expandByObject(child); // the model frames the view even while hidden
    }
  }
  if (box.isEmpty()) box.set(new THREE.Vector3(-3, 0, -3), new THREE.Vector3(3, 2.5, 3));
  const center = box.getCenter(new THREE.Vector3());
  const radius = Math.max(box.getSize(new THREE.Vector3()).length() / 2, 1);
  // from the south, above, looking at the middle: a dollhouse view
  walk.lookAt(center.clone().add(new THREE.Vector3(0, 0.9 * radius, 1.1 * radius)), center);
}

/** Index of the splat to show first: ?splat= (id, URL or none), else the catalogue's default, else the first. */
function initialSplat(defaultId: string | null | undefined): number {
  const want = params.splat;
  if (want === "none") return -1;
  if (want !== undefined) {
    const byId = entries.findIndex((e) => e.id === want || e.url === want);
    if (byId >= 0) return byId;
    entries.push({ id: want, label: want.split("/").pop() ?? want, url: want }); // any splat URL
    return entries.length - 1;
  }
  const byDefault = entries.findIndex((e) => e.id === defaultId);
  return byDefault >= 0 ? byDefault : entries.length ? 0 : -1;
}

/** Stand at a named view: its position and direction (walkable from there; a still's slight roll is
 * dropped) and its field of view. The URL gets ?view= so a reload comes back to it. */
function goToView(index: number): void {
  const v = views[index];
  if (!v) return;
  const m = v.pose; // row-major camera-to-world, OpenGL axes: the camera looks along -z (third column)
  const eye: V3 = [m[3], m[7], m[11]];
  const ahead: V3 = [eye[0] - m[2], eye[1] - m[6], eye[2] - m[10]];
  frozen = false;
  camera.fov = v.fov;
  camera.updateProjectionMatrix();
  walk.lookAt(sceneToThree(eye), sceneToThree(ahead));
  const url = new URL(window.location.href);
  for (const key of ["pose", "cam", "look"]) url.searchParams.delete(key);
  url.searchParams.set("view", v.name);
  url.searchParams.set("fov", String(v.fov));
  window.history.replaceState(null, "", url.toString().replace(/%2C/gi, ","));
  panel?.showView(index);
  status("pose", `view: ${v.name}${v.note ? ` · ${v.note}` : ""}`);
}

async function main(): Promise<void> {
  const choice = await loadScans(params.scan);
  scan = choice.current;
  if (choice.error) status("scan", choice.error, true);
  else if (scan) status("scan", `scan: ${scan.label ?? scan.id}`);
  const [catalog, glbUrl, namedViews] = await Promise.all([
    loadCatalog(choice.base),
    firstExisting(params.glb ?? [`${choice.base}model.glb`]),
    loadViews(choice.base),
  ]);
  entries = catalog.splats;
  views = namedViews;
  const first = initialSplat(catalog.default);
  if (!glbUrl && params.explicitGlb && params.glb?.length) status("glb", `glb not found: ${params.glb[0]}`, true);
  if (first < 0 && !glbUrl) status("splat", "nothing to show: run scanmodel export, or pass ?splat=…&glb=…", true);

  panel = new Panel(entries, {
    selectScan: (id) => {
      if (id !== scan?.id) window.location.href = scanHref(window.location.href, id);
    },
    selectView: (i) => goToView(i),
    selectSplat: (i) => void showSplat(i),
    setModel,
    setOpacity: (v) => {
      modelOpacity = v;
      styleModel();
    },
    setWireframe: (on) => {
      modelWireframe = on;
      styleModel();
      panel?.showModel(model?.visible ?? false, modelOpacity, modelWireframe);
    },
    copyView: (walkable) => void copyView(walkable),
  }, glbUrl !== undefined, choice.scans, scan?.id, views); // prettier-ignore

  const jobs: Promise<void>[] = [showSplat(first)];
  if (glbUrl) jobs.push(loadModel(glbUrl, params.glbUp).catch((e: unknown) => status("glb", `model ${glbUrl}: ${String(e)}`, true)));
  await Promise.all(jobs);
  setModel(params.model ?? first < 0); // default: the model only when there is no splat to show

  if (params.pose) {
    setPose(params.pose);
    status("pose", "camera fixed by ?pose (shift+C copies a walkable URL of this view)");
  } else if (params.cam) {
    const look = params.look ?? [params.cam[0], params.cam[1] + 1, params.cam[2]]; // default: face north
    walk.lookAt(sceneToThree(params.cam), sceneToThree(look));
  } else if (params.view && views.some((v) => v.name === params.view)) {
    goToView(views.findIndex((v) => v.name === params.view));
  } else {
    if (params.view) status("pose", `no named view ${params.view} in this scan`, true);
    frameAll();
  }
}

// --- copy view: a URL that reopens this exact view, and that `scanmodel view add` takes -----------------------
const toastEl = document.getElementById("toast")!;
let toastTimer = 0;
function toast(msg: string): void {
  toastEl.textContent = msg;
  toastEl.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = window.setTimeout(() => toastEl.classList.remove("show"), 1800);
}

async function writeClipboard(text: string): Promise<boolean> {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    // no clipboard API or no permission (e.g. the page lost focus): the old way
    const area = document.createElement("textarea");
    area.value = text;
    document.body.appendChild(area);
    area.select();
    const ok = document.execCommand("copy");
    area.remove();
    return ok;
  }
}

/** Copies the current view: exact `?pose=` (opens frozen), or walkable `?cam=&look=` (yaw and pitch only, as
 * walking never rolls), with `?fov=` and the shown layers; the page's other parameters stay. */
async function copyView(walkable: boolean): Promise<string> {
  const where = walkable ? { cam: threeToScene(camera.position), look: threeToScene(walk.target()) } : { pose: getPose() };
  const url = viewHref(window.location.href, {
    ...where,
    fov: camera.fov,
    splat: entries[splatIndex]?.id ?? "none",
    model: model ? model.visible : undefined,
    scan: scan?.id, // the pose means something in this scan's frame only
  });
  console.log(`view URL (${walkable ? "walkable" : "exact"}): ${url}`);
  const copied = await writeClipboard(url);
  toast(copied ? `View URL copied${walkable ? " (walkable)" : ""}` : "Copy failed: the URL is in the console");
  return url;
}
(window as unknown as { copyView: typeof copyView }).copyView = copyView;

// --- loop ----------------------------------------------------------------------------------------
window.addEventListener("resize", () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});
window.addEventListener("keydown", (e) => {
  if (e.target instanceof HTMLInputElement || e.target instanceof HTMLSelectElement || e.metaKey || e.ctrlKey) return;
  if (e.code === "KeyG") helpers.visible = !helpers.visible;
  if (e.code === "KeyF") frameAll();
  if (e.code === "KeyM" && model) setModel(!model.visible);
  if (e.code === "KeyX" && model) {
    modelWireframe = !modelWireframe;
    styleModel();
    panel?.showModel(model.visible, modelOpacity, modelWireframe);
  }
  if (e.code === "KeyP") panel?.toggle();
  if (e.code === "KeyC" && !e.repeat && !e.altKey) void copyView(e.shiftKey);
  const digit = /^Digit([0-9])$/.exec(e.code);
  if (digit) {
    const i = Number(digit[1]) - 1; // 0: none, 1..9: the splats in panel order
    if (i < entries.length && i !== splatIndex) void showSplat(i);
  }
});

const fmt = (v: V3) => v.map((c) => c.toFixed(2)).join(", ");
const timer = new THREE.Timer();
timer.connect(document);
let poseTimer = 0;
renderer.setAnimationLoop((time) => {
  timer.update(time);
  const dt = Math.min(timer.getDelta(), 0.1);
  if (!frozen) walk.update(dt);
  renderer.render(scene, camera);
  poseTimer += dt;
  if (poseTimer > 0.2) {
    poseTimer = 0;
    // scene-frame coordinates, ready to paste into ?cam=…&look=… or a scene `views` entry
    poseEl.textContent = `cam ${fmt(threeToScene(camera.position))} · look ${fmt(threeToScene(walk.target()))}`;
  }
});

main().catch((e: unknown) => status("main", String(e), true));
