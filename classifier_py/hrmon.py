"""
The board (HRMon sketch) sends 10 IMU samples per packet at 50 Hz

Commands
  train <still|walking|running> <seconds> : to record user-labeled data, and refit
  undo : remove the last recording and refit
  refit : recompute thresholds from all data now
  status : show thresholds and recordings
  quit
  python live.py --refit     recompute thresholds without connecting to the board
"""
import sys, os, glob, json, struct, time, asyncio, threading
from collections import deque
import numpy as np

NAME      = "HRMon"
DATA_UUID = "8e3f0002-6c1a-4b7e-9d52-1f0a3c5e7b90"      # IMU notifications
CMD_UUID  = "8e3f0003-6c1a-4b7e-9d52-1f0a3c5e7b90"      # commands to the board
CMD_REST, CMD_CANCEL = 0x01, 0x02

# packed struct ImuPacket on the board: uint16 seq, then 10 samples x 6 int16
PACKET_FMT = "<H60h"
PACKET_BYTES = struct.calcsize(PACKET_FMT)               # 122 bytes
SAMPLES_PER_PACKET = 10

FS = 50 # samples per second (IMU_RATE_HZ on the board)
WIN_S, STEP_S = 4, 1 # classify the last 4 s, once per second
SETTLE_S = 3 # stillness needed before "stopped" is confirmed
CHANGE_S = 2 # seconds a new active state must persist to be confirmed
MIN_ACTIVE_S = 15 # active time needed for a stop to count as the end of a bout
RECOVERY_S = 60 # the board's recovery window; an abort after this would erase its result
TRIGGER_FROM = {"walking", "running"} # use {"running"} to trigger after running only
STOP_LAG_S = SETTLE_S + 2 # estimated seconds between the actual stop and the confirmation (7-8 s in tests)
ORDER = ["still", "walking", "running"]
BASE = os.path.dirname(os.path.abspath(__file__))
DATASET = os.path.join(BASE, "dataset_live") # user data
PUBLIC_DIR = os.path.join(BASE, "dataset_public") # public data
THRESH_FILE = os.path.join(BASE, "thresholds_live.json")
OWN_SHARE = 0.5 # share of a class's weight given to your own windows once they exist
DEFAULT_THRESH = (0.104, 0.220) # starting point (from the test recordings); replaced by 'train'


def mad(w):
    return float(np.abs(w - w.mean()).mean())

def mad_windows(x):
    win, step = FS * WIN_S, FS * STEP_S
    return np.array([mad(x[i:i + win]) for i in range(0, len(x) - win + 1, step)])

def best_cut(lo, wlo, hi, whi):
    """Cutoff between a lower and a higher class, minimizing the average per-class
    weighted error (weights sum to 1 within each class). Ties resolve to the middle
    of the best range, which is the centre of the gap when the classes don't overlap."""
    olo, ohi = np.argsort(lo), np.argsort(hi)
    lo_s, hi_s = lo[olo], hi[ohi]
    clo = np.concatenate([[0.0], np.cumsum(wlo[olo] / wlo.sum())])
    chi = np.concatenate([[0.0], np.cumsum(whi[ohi] / whi.sum())])
    vals = np.unique(np.concatenate([lo, hi]))
    cands = (vals[:-1] + vals[1:]) / 2
    err = ((1 - clo[np.searchsorted(lo_s, cands, side="left")]) # lower class at/above the cut
           + chi[np.searchsorted(hi_s, cands, side="left")]) / 2 # higher class below the cut
    best = cands[err <= err.min() + 1e-12]
    return float((best[0] + best[-1]) / 2), float(err.min())


def class_windows(folder, label):
    arrs = []
    for f in sorted(glob.glob(os.path.join(folder, label, "*.csv"))):
        w = mad_windows(np.loadtxt(f, delimiter=",", skiprows=1, ndmin=1))
        if len(w):
            arrs.append(w)
    return np.concatenate(arrs) if arrs else np.array([])


def blend(pub, own):
    """Windows and weights for one class. Without own data, public windows count fully;
    with it, your windows carry OWN_SHARE of the total weight."""
    share = 0.0 if len(own) == 0 else (1.0 if len(pub) == 0 else OWN_SHARE)
    x = np.concatenate([pub, own])
    w = np.concatenate([np.full(len(pub), (1 - share) / max(len(pub), 1)),
                        np.full(len(own), share / max(len(own), 1))])
    return x, w


class Processor:
    def __init__(self):
        self.t_walk, self.t_run = self._load_thresholds()
        self.buf = deque(maxlen=FS * WIN_S)
        self.n = 0                       # samples received
        self.reset_state()
        self.rec = None
        self.last_file = None
        if not os.path.exists(THRESH_FILE) and glob.glob(os.path.join(PUBLIC_DIR, "*", "*.csv")):
            print(self.refit())

    def reset_state(self):
        self.state = None # confirmed activity
        self.bout = set() # active states seen in the current bout
        self.raw_prev, self.streak = None, 0
        self.active_secs = 0
        self.rest_t = None

    @staticmethod
    def _load_thresholds():
        if os.path.exists(THRESH_FILE):
            d = json.load(open(THRESH_FILE))
            return d["T_WALK"], d["T_RUN"]
        return DEFAULT_THRESH

    def classify(self, m):
        return "still" if m < self.t_walk else ("walking" if m < self.t_run else "running")

    # streaming 
    def feed(self, samples):
        events = []
        for s in samples:
            self.buf.append(s)
            self.n += 1
            if self.rec is not None:
                self._record(s, events)
            if self.n % (FS * STEP_S) == 0 and len(self.buf) == self.buf.maxlen:
                events += self._tick()
        return events

    def _tick(self):
        if self.rec is not None:
            return []
        events = []
        m = mad(np.fromiter(self.buf, float))
        raw = self.classify(m)
        self.streak = self.streak + 1 if raw == self.raw_prev else 1
        self.raw_prev = raw
        events.append(("TICK", m, raw))

        if self.state is None: # first window: adopt it, no events
            self.state = raw
            if raw != "still":
                self.bout.add(raw)
            return events

        now = self.n / FS
        need = SETTLE_S if raw == "still" else CHANGE_S
        if raw != self.state and self.streak >= need:
            old, self.state = self.state, raw
            events.append(("STATE", old, raw))
            if raw == "still":
                if self.active_secs >= MIN_ACTIVE_S and self.bout & TRIGGER_FROM:
                    events.append(("REST", STOP_LAG_S))
                    self.rest_t = now
                self.active_secs = 0
                self.bout = set()
            else:
                self.bout.add(raw)
                if self.rest_t is not None and now - self.rest_t < RECOVERY_S:
                    events.append(("RESUMED",))
                self.rest_t = None
        if self.state != "still":
            self.active_secs += 1
        return events

    # Record/Adapt
    def start_recording(self, label, secs):
        self.reset_state()
        self.rec = dict(label=label, secs=secs, skip=FS * 5, data=[])
        return f"get into position - recording '{label}' for {secs} s starts in 5 s"

    def _record(self, s, events):
        r = self.rec
        if r["skip"] > 0:
            r["skip"] -= 1
            if r["skip"] == 0:
                events.append(("INFO", f"recording '{r['label']}' - go"))
            return
        r["data"].append(s)
        if len(r["data"]) >= r["secs"] * FS:
            self.rec = None
            self.reset_state()
            events.append(("INFO", self._finish(r)))

    def _finish(self, r):
        folder = os.path.join(DATASET, r["label"])
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, f"{r['label']}_{len(glob.glob(folder + '/*.csv')):03d}.csv")
        x = np.array(r["data"])
        np.savetxt(path, x, delimiter=",", header="mag_g", comments="", fmt="%.4f")
        self.last_file = path
        w = mad_windows(x)
        agree = float(np.mean([self.classify(m) == r["label"] for m in w])) if len(w) else 0.0
        msg = [f"saved {path} ({len(x)} samples)"]
        if agree < 0.7:
            msg.append(f"WARNING: only {agree:.0%} of this recording looks like '{r['label']}' "
                       "under the current thresholds. Mislabeled? Type 'undo' to remove it.")
        msg.append(self.refit())
        return "\n".join(msg)

    def refit(self):
        data, counts = {}, {}
        for c in ORDER:
            pub, own = class_windows(PUBLIC_DIR, c), class_windows(DATASET, c)
            counts[c] = (len(pub), len(own))
            data[c] = blend(pub, own) if len(pub) + len(own) else None
        missing = [c for c in ORDER if data[c] is None]
        if missing:
            return (f"thresholds unchanged (no data yet for: {', '.join(missing)}; "
                    f"run import_pamap2.py or record with 'train')")
        self.t_walk, _ = best_cut(*data["still"], *data["walking"])
        self.t_run, _ = best_cut(*data["walking"], *data["running"])
        json.dump({"T_WALK": round(self.t_walk, 4), "T_RUN": round(self.t_run, 4),
                   "windows": {c: {"public": counts[c][0], "own": counts[c][1]} for c in ORDER}},
                  open(THRESH_FILE, "w"), indent=2)
        return (f"thresholds updated: T_WALK={self.t_walk:.3f}  T_RUN={self.t_run:.3f}  "
                f"(public/own windows: {', '.join(f'{c} {counts[c][0]}/{counts[c][1]}' for c in ORDER)})")

    def undo(self):
        if not self.last_file or not os.path.exists(self.last_file):
            return "nothing to undo"
        os.remove(self.last_file)
        gone, self.last_file = self.last_file, None
        return f"removed {gone}\n" + self.refit()

    def status(self):
        own = {c: len(glob.glob(os.path.join(DATASET, c, "*.csv"))) for c in ORDER}
        pub = {c: len(glob.glob(os.path.join(PUBLIC_DIR, c, "*.csv"))) for c in ORDER}
        return (f"T_WALK={self.t_walk:.3f}  T_RUN={self.t_run:.3f}  "
                f"own recordings: {own}  public recordings: {pub}  confirmed state: {self.state}")


#  Bluetooth 
def parse_packet(data):
    """One notification -> (sequence number, 10 acceleration magnitudes in g)."""
    seq, *raw = struct.unpack(PACKET_FMT, bytes(data))
    s = np.array(raw, dtype=float).reshape(SAMPLES_PER_PACKET, 6)
    acc = s[:, :3] / 1000.0               # mg -> g
    # gyro (s[:, 3:] / 10.0, deg/s) is in the packet but not used yet
    return seq, np.linalg.norm(acc, axis=1).tolist()


def console(loop, q):
    for line in sys.stdin:
        loop.call_soon_threadsafe(q.put_nowait, line.strip())


def handle_cmd(proc, line):
    p = line.split()
    if not p:
        return False
    if p[0] == "train" and len(p) == 3 and p[1] in ORDER and p[2].isdigit() and int(p[2]) >= 8:
        print(proc.start_recording(p[1], int(p[2])))
    elif p[0] == "undo":
        print(proc.undo())
    elif p[0] == "refit":
        print(proc.refit())
    elif p[0] == "status":
        print(proc.status())
    elif p[0] == "quit":
        return True
    else:
        print("commands: train <still|walking|running> <seconds >= 8>, undo, refit, status, quit")
    return False


async def run_ble(proc):
    from bleak import BleakScanner, BleakClient # imported here so Processor can be tested without bleak
    print(f"Scanning for '{NAME}'...")
    dev = await BleakScanner.find_device_by_name(NAME, timeout=15)
    if dev is None:
        print("Board not found. Is it powered, advertising, and not connected to a phone?")
        return

    loop = asyncio.get_running_loop()
    events_q, cmd_q = asyncio.Queue(), asyncio.Queue()
    link = {"seq": None, "val": 1.0, "drops": 0, "warned": False}

    def on_data(_, data):
        if len(data) != PACKET_BYTES:
            if not link["warned"]:
                print(f"WARNING: got a {len(data)}-byte packet, expected {PACKET_BYTES}. "
                      "The Bluetooth MTU may be too small; try reconnecting.")
                link["warned"] = True
            return
        seq, vals = parse_packet(data)
        if link["seq"] is not None:
            gap = (seq - link["seq"] - 1) % 65536
            if 0 < gap < 500:                        # keep timing aligned across lost packets
                link["drops"] += gap
                vals = [link["val"]] * (SAMPLES_PER_PACKET * gap) + vals
        link["seq"], link["val"] = seq, vals[-1]
        for ev in proc.feed(vals):
            events_q.put_nowait(ev)

    async with BleakClient(dev) as client:
        await client.start_notify(DATA_UUID, on_data)
        print("Connected. Type: train <still|walking|running> <seconds>, undo, refit, status, quit")
        print(f"Using T_WALK={proc.t_walk:.3f}, T_RUN={proc.t_run:.3f}")
        threading.Thread(target=console, args=(loop, cmd_q), daemon=True).start()

        while client.is_connected:
            while not cmd_q.empty():
                if handle_cmd(proc, cmd_q.get_nowait()):
                    return
            try:
                ev = await asyncio.wait_for(events_q.get(), 0.2)
            except asyncio.TimeoutError:
                continue
            kind = ev[0]
            if kind == "TICK":
                print(f"{time.strftime('%H:%M:%S')}  mad={ev[1]:.3f}  {ev[2]:8s} "
                      f"confirmed={proc.state}  drops={link['drops']}")
            elif kind == "STATE":
                print(f"*** state change: {ev[1]} -> {ev[2]}")
            elif kind == "REST":
                print(f"*** REST detected - sending command to board (stopped ~{ev[1]} s ago)")
                await client.write_gatt_char(CMD_UUID, bytes([CMD_REST, ev[1]]), response=True)
            elif kind == "RESUMED":
                print("*** activity resumed during recovery - sending abort to board")
                await client.write_gatt_char(CMD_UUID, bytes([CMD_CANCEL, 0]), response=True)
            elif kind == "INFO":
                print(ev[1])
        print("Disconnected.")


if __name__ == "__main__":
    if "--refit" in sys.argv:
        print(Processor().refit())
    else:
        asyncio.run(run_ble(Processor()))