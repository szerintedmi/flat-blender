import * as THREE from "three";

/**
 * Game-style first-person navigation. Click the view to capture the mouse (Esc releases it); the mouse
 * then looks around. Arrows: up/down walk forward/back, left/right turn. Shift + arrows: up/down rise and
 * sink, left/right strafe. WASD walk and strafe, Q/E sink/rise; the wheel steps forward/back.
 * Walking is horizontal (yaw only), so looking down while walking does not sink you into the floor.
 */
export class WalkControls {
  speed = 1.5; // m/s
  turnSpeed = Math.PI / 2; // rad/s, left/right arrows
  lookSpeed = 0.0025; // rad per pixel of mouse movement
  wheelStep = 0.002; // m per wheel delta unit
  private yaw = 0; // about world up; 0 looks along -z (north in the scene frame)
  private pitch = 0;
  private readonly down = new Set<string>();
  private readonly forward = new THREE.Vector3();
  private readonly right = new THREE.Vector3();
  private readonly move = new THREE.Vector3();
  private static readonly UP = new THREE.Vector3(0, 1, 0);
  private static readonly KEYS = new Set([
    "ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight", "KeyW", "KeyA", "KeyS", "KeyD", "KeyQ", "KeyE",
  ]); // prettier-ignore

  constructor(
    private readonly camera: THREE.PerspectiveCamera,
    private readonly element: HTMLElement,
  ) {
    camera.rotation.order = "YXZ"; // yaw, then pitch: no roll creeps in
    element.addEventListener("click", () => {
      if (document.pointerLockElement !== element) void element.requestPointerLock();
    });
    document.addEventListener("mousemove", (e) => {
      if (document.pointerLockElement !== element) return;
      this.yaw -= e.movementX * this.lookSpeed;
      this.pitch -= e.movementY * this.lookSpeed;
      this.apply();
    });
    element.addEventListener(
      "wheel",
      (e) => {
        e.preventDefault();
        this.horizontalAxes();
        this.camera.position.addScaledVector(this.forward, -e.deltaY * this.wheelStep);
      },
      { passive: false },
    );
    window.addEventListener("keydown", (e) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLSelectElement || e.metaKey || e.ctrlKey) return;
      if (WalkControls.KEYS.has(e.code)) e.preventDefault();
      this.down.add(e.code);
    });
    window.addEventListener("keyup", (e) => this.down.delete(e.code));
    window.addEventListener("blur", () => this.down.clear());
  }

  get locked(): boolean {
    return document.pointerLockElement === this.element;
  }

  /** Stand at `position` and look at `target` (three.js coordinates). */
  lookAt(position: THREE.Vector3, target: THREE.Vector3): void {
    this.camera.position.copy(position);
    const d = target.clone().sub(position);
    if (d.lengthSq() === 0) return;
    this.yaw = Math.atan2(-d.x, -d.z);
    this.pitch = Math.atan2(d.y, Math.hypot(d.x, d.z));
    this.apply();
  }

  /** A point 1 m ahead of the camera, for the HUD's `look`. */
  target(out = new THREE.Vector3()): THREE.Vector3 {
    return this.camera.getWorldDirection(out).add(this.camera.position);
  }

  update(dt: number): void {
    const k = (code: string) => (this.down.has(code) ? 1 : 0);
    const shift = this.down.has("ShiftLeft") || this.down.has("ShiftRight");
    const arrowsFb = k("ArrowUp") - k("ArrowDown");
    const arrowsLr = k("ArrowRight") - k("ArrowLeft");
    const forward = (shift ? 0 : arrowsFb) + k("KeyW") - k("KeyS");
    const strafe = (shift ? arrowsLr : 0) + k("KeyD") - k("KeyA");
    const rise = (shift ? arrowsFb : 0) + k("KeyE") - k("KeyQ");
    const turn = shift ? 0 : arrowsLr;

    if (turn) {
      this.yaw -= turn * this.turnSpeed * dt;
      this.apply();
    }
    if (!forward && !strafe && !rise) return;
    this.horizontalAxes();
    this.move
      .set(0, 0, 0)
      .addScaledVector(this.forward, forward)
      .addScaledVector(this.right, strafe)
      .addScaledVector(WalkControls.UP, rise);
    if (this.move.lengthSq() === 0) return;
    this.camera.position.addScaledVector(this.move.normalize(), this.speed * dt);
  }

  private horizontalAxes(): void {
    this.forward.set(-Math.sin(this.yaw), 0, -Math.cos(this.yaw));
    this.right.set(Math.cos(this.yaw), 0, -Math.sin(this.yaw));
  }

  private apply(): void {
    const limit = Math.PI / 2 - 0.01;
    this.pitch = Math.max(-limit, Math.min(limit, this.pitch));
    this.camera.rotation.set(this.pitch, this.yaw, 0);
  }
}
