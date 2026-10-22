import numpy as np

class LineCrossTrigger:
    def __init__(self, pos=0.5, axis=0, direction=0):
        """pos: line position as fraction of frame width (axis=0) or height (axis=1).
        direction: +1 / -1 to fire only one way, 0 for either."""
        self.pos, self.axis, self.direction = pos, axis, direction
        self.last = {}
        self.fired = set()

    def __call__(self, tid, box, frame_shape):
        h, w = frame_shape[:2]
        line = self.pos * (w if self.axis == 0 else h)
        c = (box[0] + box[2]) / 2 if self.axis == 0 else (box[1] + box[3]) / 2
        prev = self.last.get(tid)
        self.last[tid] = c
        if prev is None or tid in self.fired:
            return False
        crossed = (prev - line) * (c - line) < 0
        if crossed and (self.direction == 0 or np.sign(c - prev) == self.direction):
            self.fired.add(tid)
            return True
        return False

    def prune(self, alive_ids):
        alive = set(alive_ids)
        self.last = {k: v for k, v in self.last.items() if k in alive}
        if len(self.fired) > 100:           # bound memory on long runs
            self.fired = {k for k in self.fired if k in alive}
