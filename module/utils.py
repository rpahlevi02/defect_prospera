
def pad_box(frame_shape, box, pad=0.1):
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = box
    px, py = (x2 - x1) * pad, (y2 - y1) * pad          # margin = 10% of box width / height
    return (max(int(x1 - px), 0), max(int(y1 - py), 0),
            min(int(x2 + px), w), min(int(y2 + py), h))

def crop_box(frame, box, pad=0.1):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    px, py = (x2 - x1) * pad, (y2 - y1) * pad
    x1, y1 = max(int(x1 - px), 0), max(int(y1 - py), 0)
    x2, y2 = min(int(x2 + px), w), min(int(y2 + py), h)
    return frame[y1:y2, x1:x2]

def masked_crop(frame, mask, box, pad=0.1, bg=0):
    x1, y1, x2, y2 = pad_box(frame.shape, box, pad)
    crop = frame[y1:y2, x1:x2].copy()
    crop[~mask[y1:y2, x1:x2]] = bg
    return crop