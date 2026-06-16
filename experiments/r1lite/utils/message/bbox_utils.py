import cv2 as cv
import numpy as np


def simple_visual_bbox(image_array, bbox):
    x1, y1, x2, y2 = bbox
    vis_image = image_array.copy()
    cv.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv.imwrite("debug_bbox.jpg", vis_image)

def get_paligemma_box_instruction(image, bbox, target_image_size=224, scale=1024):
    bbox = np.array(bbox)
    h, w  = image.shape[:2]
    h_scale, w_scale = target_image_size / h, target_image_size / w
    bbox = bbox * np.array([w_scale, h_scale, w_scale, h_scale])
    image = cv.resize(image, (target_image_size, target_image_size))
    simple_visual_bbox(cv.cvtColor(image, cv.COLOR_RGB2BGR), bbox) # simple resize for visualization here
    bbox = np.clip(np.round(bbox / target_image_size * scale), 0, scale - 1).astype(np.int32)
    rel_x1, rel_y1, rel_x2, rel_y2 = bbox
    y_min = min(rel_y1, rel_y2)
    x_min = min(rel_x1, rel_x2)
    y_max = max(rel_y1, rel_y2)
    x_max = max(rel_x1, rel_x2)
    bbox = f"<loc{y_min}><loc{x_min}><loc{y_max}><loc{x_max}>"
    return bbox


def get_bbox_image(rgb_head_image:np.ndarray, 
                   bbox, target_height=224, target_width=224):
    """
    Extract the bbox region from the image and resize it to the target size.
    
    Args:
        rgb_head_image: RGB image array, shape (H, W, 3).
        bbox: bounding box [x1, y1, x2, y2].
        target_height: target height.
        target_width: target width.
        
    Returns:
        Resized image array, shape (target_height, target_width, 3), dtype=uint8.
    """
    # Ensure the image is float32.
    rgb_head_image = rgb_head_image.astype(np.float32)
    H, W, _ = rgb_head_image.shape

    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    side = max(bw, bh)  # Use the longest bbox side as the square side length.
    cx, cy = x1 + bw / 2, y1 + bh / 2

    # Compute the square bbox.
    new_x1 = int(np.floor(cx - side / 2))
    new_y1 = int(np.floor(cy - side / 2))
    new_x2 = int(np.ceil(cx + side / 2))
    new_y2 = int(np.ceil(cy + side / 2))

    # Compute required padding when the box exceeds image boundaries.
    pad_left = max(0, -new_x1)
    pad_top = max(0, -new_y1)
    pad_right = max(0, new_x2 - W)
    pad_bottom = max(0, new_y2 - H)

    # Pad with OpenCV.
    if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
        img_padded = cv.copyMakeBorder(
            rgb_head_image,
            top=pad_top,
            bottom=pad_bottom,
            left=pad_left,
            right=pad_right,
            borderType=cv.BORDER_CONSTANT,
            value=0
        )
    else:
        img_padded = rgb_head_image

    # Update crop coordinates with padding offsets.
    crop_x1 = new_x1 + pad_left
    crop_y1 = new_y1 + pad_top
    crop_x2 = new_x2 + pad_left
    crop_y2 = new_y2 + pad_top

    # Crop the square region.
    crop = img_padded[crop_y1:crop_y2, crop_x1:crop_x2, :]
    
    # Resize with OpenCV using bilinear interpolation.
    crop_resized = cv.resize(
        crop, 
        (target_width, target_height), 
        interpolation=cv.INTER_LINEAR
    )
    
    # Convert to uint8.
    crop_resized = np.clip(crop_resized, 0, 255).astype(np.uint8)

    # Debug output.
    cv.imwrite("debug_condition_image.png",
               cv.cvtColor(crop_resized, cv.COLOR_RGB2BGR))
    
    return crop_resized.transpose(2, 0, 1)
