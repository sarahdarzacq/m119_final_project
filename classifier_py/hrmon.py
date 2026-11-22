"""
The board (HRMon sketch) sends 10 IMU samples per packet at 50 Hz

Commands
    train <still|walking|running> <seconds> : record and add labeled training data
    undo : remove the last recording and reload the dataset
    why : explain the latest classification
    status : show classifier and dataset information
    quit
"""
import sys, os, glob, struct, time, asyncio, threading
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

K = 15 # neighbor threshold

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, "dataset")
OWN_PREFIX = "own_"
OWN_SHARE = 0.5 # share of a class's weight given to your own windows once they exist

def mad(w):
    return float(np.abs(w - w.mean()).mean())

def window_features(w):
    w = np.asarray(w, dtype=float)
    w = w - w.mean()
    n = len(w)
    p = np.abs(np.fft.rfft(w * np.hanning(n+1)[:-1])) ** 2
    f = np.fft.rfftfreq(n, 1 / FS)
    m = (f > 0.5) & (f < 4)
    return np.array([np.abs(w).mean(), w.std(), np.ptp(w), 
                     f[m][p[m].argmax()], p[m].max() / (p[m].sum() + 1e-12)])

def feature_windows(x):
    win, step = FS * WIN_S, FS * STEP_S
    rows = [window_features(x[i:i + win]) for i in range(0, len(x) - win + 1, step)]
    return np.array(rows) if rows else np.empty((0, 5))

def is_own(path):
    return os.path.basename(path).startswith(OWN_PREFIX)

def dataset_files(label, own=None):
    return [f for f in sorted(glob.glob(os.path.join(DATA_DIR, label, "*.csv"))) if own is None or is_own(f) == own]

class KnnStore:
    def __init__(self):
        self.X = np.empty((0, 5))
        self.yi = np.empty(0, int)
        self.own = np.empty(0, bool)
        self._prepare()

    def load(self):
        Xs, ys, owns = [], [] ,[]
        for ci, c in enumerate(ORDER):
            for f in dataset_files(c):
                F = feature_windows(np.loadtxt(f, delimiter=",", skiprows=1, ndmin=1))
                if len(F):
                    Xs.append(F)
                    ys.append(np.full(len(F), ci))
                    owns.append(np.full(len(F), is_own(f)))
        self.X = np.vstack(Xs) if Xs else np.empty((0,5))
        self.yi = np.concatenate(ys) if ys else np.empty(0, int)
        self.own = np.concatenate(owns) if owns else np.empty(0, bool)
        self._prepare()

    def add(self, F, label, own):
        if len(F) == 0: return 0
        self.X = np.vstack([self.X, F])
        self.yi = np.concatenate([self.yi, np.full(len(F), ORDER.index(label))])
        self.own = np.concatenate([self.own, np.full(len(F), own)])
        self._prepare()
        return len(F)

    def _prepare(self):
        if len(self.X) == 0:
            self.mu, self.sd, self.Z, self.w = np.zeros(5), np.ones(5), np.empty((0, 5)), np.empty(0)
            return
        self.mu, self.sd = self.X.mean(0), self.X.std(0) + 1e-9
        self.Z = (self.X - self.mu) / self.sd
        self.w = np.zeros(len(self.yi))
        for ci in range(len(ORDER)):
            m = self.yi == ci
            mo, mp = m & self.own, m & ~self.own
            share = 0.0 if not mo.any() else (1.0 if not mp.any() else OWN_SHARE)
            self.w[mp] = (1 - share) / max(mp.sum(), 1)
            self.w[mo] = share / max(mo.sum(), 1)

    def ready(self): return all((self.yi == i).any() for i in range(len(ORDER)))

    def summary(self):
        return ", ".join(f"{c} {int(((self.yi == i) & ~self.own).sum())}+{int(((self.yi == i) & self.own).sum())}"
                         for i, c in enumerate(ORDER)) + " windows (public+own)"

    def classify(self, F):
        d = np.linalg.norm(self.Z - (F - self.mu) / self.sd, axis=1)
        idx = np.argpartition(d, K)[:K] if len(d) > K else np.arange(len(d))
        votes = np.zeros(len(ORDER))
        np.add.at(votes, self.yi[idx], self.w[idx] / (d[idx] + 1e-6))
        k = int(votes.argmax())
        conf = float(votes[k] / votes.sum())
        return ORDER[k], conf, {"features": F, "conf": conf, "nearest": float(d.min()),
                                "neighbors": np.bincount(self.yi[idx], minlength=len(ORDER)),
                                "n_own": int(self.own[idx].sum())}

class Processor:
    def __init__(self):
        self.buf = deque(maxlen=FS * WIN_S)
        self.n = 0 # samples received
        self.reset_state()
        self.rec = None
        self.last_file = None
        self.last_info = None
        self.store = KnnStore()
        self.store.load()
        if not self.store.ready():
            raise RuntimeError(
                "kNN requires training data for still, walking, and running."
            )

    def reset_state(self):
        self.state = None # confirmed activity
        self.bout = set() # active states seen in the current bout
        self.raw_prev, self.streak = None, 0
        self.active_secs = 0
        self.rest_t = None

    def _decide(self, F):
        return self.store.classify(F)

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
        F = window_features(np.asarray(self.buf, dtype=float))
        raw, conf, info = self._decide(F)
        self.last_info = info
        self.streak = self.streak + 1 if raw == self.raw_prev else 1
        self.raw_prev = raw
        events.append(("TICK", float(F[0]), raw, conf))

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
        folder = os.path.join(DATA_DIR, r["label"])
        os.makedirs(folder, exist_ok=True)
        n = 0
        while os.path.exists(os.path.join(folder, f"{OWN_PREFIX}{r['label']}_{n:03d}.csv")):
            n += 1
        path = os.path.join(folder, f"{OWN_PREFIX}{r['label']}_{n:03d}.csv")
        x = np.array(r["data"])
        np.savetxt(path, x, delimiter=",", header="mag_g", comments="", fmt="%.4f")
        self.last_file = path
        F = feature_windows(x)
        msg = [f"saved {path} ({len(x)} samples, {len(F)} windows)"]
        if len(F):
            # judged BEFORE adding, otherwise the new windows would vote for themselves
            agree = float(np.mean([self._decide(f)[0] == r["label"] for f in F]))
            if agree < 0.7:
                msg.append(f"NOTE: only {agree:.0%} of this recording looks like '{r['label']}' to the "
                           f"current knn classifier. That is expected if your motion differs from the "
                           "dataset (it adapts now); if you picked the wrong label, type 'undo'.")
        self.store.add(F, r["label"], own=True)
        msg.append(f"added to the dataset: {self.store.summary()}")
        return "\n".join(msg)
    
    def undo(self):
        if not self.last_file or not os.path.exists(self.last_file): return "nothing to undo"

        os.remove(self.last_file)
        gone = self.last_file
        self.last_file = None

        self.store.load()
        return f"removed {gone}\n{self.store.summary()}"
    
    def status(self):
        return (
            f"classifier=kNN  "
            f"dataset: {self.store.summary()}  "
            f"confirmed state: {self.state}"
        )
    
    def why(self):
        i = self.last_info
        if not i:
            return "nothing classified yet (the first window needs 4 s of data)"
        F = i["features"]
        head = (f"features: mad={F[0]:.3f} g, spread={F[1]:.3f}, range={F[2]:.2f}, "
                f"cadence={F[3]:.2f} Hz, peak share={F[4]:.2f}")
        votes = ", ".join(f"{n} {c}" for c, n in zip(ORDER, i["neighbors"]) if n)
        return (head + f"\n{K} nearest stored windows: {votes} ({i['n_own']} from your recordings); "
                       f"nearest distance {i['nearest']:.2f} (large = unlike anything stored)"
                       f"\nvote -> {self.raw_prev} (confidence {i['conf']:.0%}); confirmed state: {self.state}")

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
    elif p[0] == "why":
        print(proc.why())
    elif p[0] == "status":
        print(proc.status())
    elif p[0] == "quit":
        return True
    else:
        print("commands: train <still|walking|running> <seconds >= 8>, undo, why, "
              "status, quit")
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
            if 0 < gap < 500: # keep timing aligned across lost packets
                link["drops"] += gap
                vals = [link["val"]] * (SAMPLES_PER_PACKET * gap) + vals
        link["seq"], link["val"] = seq, vals[-1]
        for ev in proc.feed(vals):
            events_q.put_nowait(ev)

    async with BleakClient(dev) as client:
        await client.start_notify(DATA_UUID, on_data)
        print("Connected. Type: train <still|walking|running> <seconds>, undo, why, "
              "status, quit")
        print(proc.status())
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
                conf = f"  conf={ev[3]:.0%}" if ev[3] is not None else ""
                print(f"{time.strftime('%H:%M:%S')}  mad={ev[1]:.3f}  {ev[2]:8s}{conf}  "
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
    try:
        asyncio.run(run_ble(Processor()))
    except KeyboardInterrupt:
        print("\nProgram stopped by user.")