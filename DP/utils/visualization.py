import cv2

def draw_custom_annotations(frame, result, names=None, thickness=2, mode='Name'):
    """
    Function for custom visualization
    
    Args:
        frame: frame (numpy array)
        result: object results from YOLO
        names: class names dictionary (optional)
        thickness: box line thickness
        mode: 'Name' - label displays class name, 'ID' - label displays class ID
    """

    COLOR_MAP = {
        0: (49,211,0),      # Tree - green
        1: (255,255,0),     # Bush - cyan
        2: (128, 0, 128),   # Stone - purple
        3: (0,0,255),       # Object - red
        4: (203,192,255)    # Dog - pink
    }

    TEXT_COLOR_MAP = {
        0: (0,0,0),         # Tree - black
        1: (0,0,0),         # Bush
        2: (255,255,255),   # Stone - white
        3: (255,255,255),   # Object
        4: (0,0,0)          # Dog
    }   

    DEFAULT_COLOR = (255,255,255)

    if result.boxes is None:
        return frame

    boxes = result.boxes.xyxy.cpu().numpy()
    clss = result.boxes.cls.cpu().numpy()
    ids = result.boxes.id.int().cpu().numpy() if result.boxes.id is not None else None

    for i, box in enumerate(boxes):
        # get label (box) info
        x1, y1, x2, y2 = map(int, box)
        cls_id = int(clss[i])
        color = COLOR_MAP.get(cls_id)
        text_color = TEXT_COLOR_MAP.get(cls_id)
        
        # get class name if available
        class_name = names[cls_id] if names is not None else f"Cls: {cls_id}"
        
        # mode selection
        if mode == 'Name' and names is not None:
            label = names[cls_id]
        elif mode == 'ID':
            label = f"{cls_id}"
        else:
            label = f"{cls_id}" # Fallback

        # if tracking (frame has an id), add tracking number
        if ids is not None:
            label = f"Id:{ids[i]}, Cls: {label}"

        # draw box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        # draw label
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - 20), (x1 + label_size[0], y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)
    
    return frame