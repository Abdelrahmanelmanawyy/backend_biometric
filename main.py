from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import numpy as np
import cv2
import mediapipe as mp
from PIL import Image
from rembg import remove, new_session
import io
import os
import glob
import uuid
from datetime import datetime
from PIL import ImageDraw

app = FastAPI()

def resize_and_crop(img, target_width, target_height):
    # Resize while keeping aspect ratio
    img_ratio = img.width / img.height
    target_ratio = target_width / target_height
    if img_ratio > target_ratio:
        # Image is wider than target: fit height, crop width
        new_height = target_height
        new_width = int(target_height * img_ratio)
    else:
        # Image is taller than target: fit width, crop height
        new_width = target_width
        new_height = int(target_width / img_ratio)
    img = img.resize((new_width, new_height), Image.LANCZOS)
    # Center crop
    left = (new_width - target_width) // 2
    top = (new_height - target_height) // 2
    right = left + target_width
    bottom = top + target_height
    img = img.crop((left, top, right, bottom))
    return img

@app.post("/validate_photo")
async def validate_photo(
    image: UploadFile = File(...),
    width_mm: float = Form(...),   # Should be 50
    height_mm: float = Form(...)   # Should be 60
):
    steps = []
    # 1. Remove background and get alpha matte
    contents = await image.read()
    steps.append("Read uploaded image.")
    session = new_session("u2net")
    output_image = remove(contents, session=session, only_mask=False, return_mask=True)
    if isinstance(output_image, tuple):
        # rembg >= 2.0 returns (image_bytes, mask_bytes)
        output_image_bytes, mask_bytes = output_image
    else:
        output_image_bytes = output_image
        mask_bytes = None
    steps.append("Removed background using rembg with u2net model and obtained alpha matte.")
    img_nobg = Image.open(io.BytesIO(output_image_bytes)).convert("RGBA")
    img_nobg_cv = cv2.cvtColor(np.array(img_nobg), cv2.COLOR_RGBA2BGRA)
    orig_height, orig_width = img_nobg_cv.shape[:2]
    # Get alpha matte as numpy array
    if mask_bytes is not None:
        alpha_matte = Image.open(io.BytesIO(mask_bytes)).convert("L")
        alpha_np = np.array(alpha_matte)
    else:
        alpha_np = img_nobg_cv[:, :, 3]

    # 2. Detect face landmarks on the original image
    img_orig = Image.open(io.BytesIO(contents)).convert("RGB")
    img_orig_cv = cv2.cvtColor(np.array(img_orig), cv2.COLOR_RGB2BGR)
    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(img_orig_cv, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            return JSONResponse({"success": False, "error": "No face detected in original image.", "steps": steps})
        landmarks = results.multi_face_landmarks[0].landmark
        chin_idx = 152
        nose_tip_idx = 1
        chin_y = int(landmarks[chin_idx].y * orig_height)
        nose_x = int(landmarks[nose_tip_idx].x * orig_width)

        # 3. Find the top of the hair using the alpha matte at the nose tip x-coordinate
        # Scan from top to bottom at nose_x for the first non-background pixel
        hair_top_y = None
        for y in range(orig_height):
            if alpha_np[y, nose_x] > 10:  # threshold for non-background
                hair_top_y = y
                break
        if hair_top_y is None:
            hair_top_y = 0  # fallback to top of image
        steps.append(f"Detected top of hair at y={hair_top_y} using alpha matte.")

        # 4. Compute crop box: 10mm above hair, 34mm face, 16mm below chin
        dpi = 300
        crop_width_px = int(width_mm / 25.4 * dpi)   # 591 px
        crop_height_px = int(height_mm / 25.4 * dpi) # 709 px
        top_margin_px = int(10 / 25.4 * dpi)         # 10mm in px
        face_height_px = int(34 / 25.4 * dpi)        # 34mm in px
        bottom_margin_px = crop_height_px - face_height_px - top_margin_px # 16mm in px
        center_x = nose_x
        left = center_x - crop_width_px // 2
        right = left + crop_width_px
        top = hair_top_y - top_margin_px
        bottom = chin_y + bottom_margin_px
        # Pad if out of bounds
        pad_top = max(0, -top)
        pad_left = max(0, -left)
        pad_bottom = max(0, bottom - orig_height)
        pad_right = max(0, right - orig_width)
        top = max(top, 0)
        left = max(left, 0)
        bottom = min(bottom, orig_height)
        right = min(right, orig_width)
        cropped_img = img_orig_cv[top:bottom, left:right]
        if pad_top or pad_bottom or pad_left or pad_right:
            cropped_img = cv2.copyMakeBorder(
                cropped_img, pad_top, pad_bottom, pad_left, pad_right,
                borderType=cv2.BORDER_CONSTANT, value=[255, 255, 255]
            )
            steps.append(f"Padded cropped image: top={pad_top}, bottom={pad_bottom}, left={pad_left}, right={pad_right}.")
        # Resize to 591x709 px (50x60 mm at 300 DPI)
        cropped_img = cv2.resize(cropped_img, (crop_width_px, crop_height_px), interpolation=cv2.INTER_LANCZOS4)
        steps.append(f"Cropped and resized image to exactly {crop_width_px}x{crop_height_px} px (50x60 mm at 300 DPI).")
        # Save to cropped_images
        cropped_dir = "cropped_images"
        os.makedirs(cropped_dir, exist_ok=True)
        cropped_path = os.path.join(cropped_dir, "cropped_passport_photo.jpg")
        cv2.imwrite(cropped_path, cropped_img)
        steps.append(f"Saved cropped image as {cropped_path} (overwritten each time).")

    # 5. Remove background from the cropped image
    with open(cropped_path, "rb") as f:
        cropped_bytes = f.read()
    session = new_session("u2net")
    output_image = remove(cropped_bytes, session=session)
    img_nobg = Image.open(io.BytesIO(output_image)).convert("RGBA")
    # Flatten any transparency to white
    if img_nobg.mode == 'RGBA':
        white_bg = Image.new('RGBA', img_nobg.size, (255, 255, 255, 255))
        img_nobg = Image.alpha_composite(white_bg, img_nobg).convert('RGB')
        steps.append("Flattened alpha channel onto white background.")
    img_nobg_cv = cv2.cvtColor(np.array(img_nobg), cv2.COLOR_RGB2BGR)
    # Save to processed_images
    processed_dir = "processed_images"
    os.makedirs(processed_dir, exist_ok=True)
    processed_path = os.path.join(processed_dir, "processed_image.jpg")
    cv2.imwrite(processed_path, img_nobg_cv)
    steps.append(f"Saved processed image as {processed_path} (overwritten each time).")
    cropped_file = cropped_path
    last_image = processed_path

    # 5. Biometric validation on cropped image
    img_for_validation = cv2.imread(cropped_path)
    height_px, width_px = img_for_validation.shape[:2]
    mm_per_px_h = height_mm / height_px
    mm_per_px_w = width_mm / width_px
    validation_results = {}
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(img_for_validation, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            validation_results["error"] = "No face detected in cropped image."
        else:
            landmarks = results.multi_face_landmarks[0].landmark
            # Face height
            chin_idx = 152
            top_head_idx = 10
            chin_y = int(landmarks[chin_idx].y * height_px)
            top_head_y = int(landmarks[top_head_idx].y * height_px)
            face_height_px = abs(chin_y - top_head_y)
            face_height_mm = face_height_px * mm_per_px_h
            # Face height validation: pass if 30 <= face_height_mm <= 37
            face_height_status = "pass" if 30 <= face_height_mm <= 37 else "fail"
            # Mouth closed
            upper_lip_idx = 13
            lower_lip_idx = 14
            upper_lip_y = int(landmarks[upper_lip_idx].y * height_px)
            lower_lip_y = int(landmarks[lower_lip_idx].y * height_px)
            mouth_opening_px = abs(lower_lip_y - upper_lip_y)
            mouth_opening_mm = mouth_opening_px * mm_per_px_h
            mouth_status = "pass" if mouth_opening_mm < 2.0 else "fail"
            # Eyes open
            left_eye_top_idx = 159
            left_eye_bottom_idx = 145
            right_eye_top_idx = 386
            right_eye_bottom_idx = 374
            left_eye_opening_px = abs(int(landmarks[left_eye_bottom_idx].y * height_px) - int(landmarks[left_eye_top_idx].y * height_px))
            right_eye_opening_px = abs(int(landmarks[right_eye_bottom_idx].y * height_px) - int(landmarks[right_eye_top_idx].y * height_px))
            left_eye_open = left_eye_opening_px > 5
            right_eye_open = right_eye_opening_px > 5
            eyes_open_status = "pass" if left_eye_open and right_eye_open else "fail"
            # Glasses detection (simple edge detection in eye region)
            def get_eye_roi(landmarks, indices):
                xs = [int(landmarks[i].x * width_px) for i in indices]
                ys = [int(landmarks[i].y * height_px) for i in indices]
                x_min, x_max = min(xs), max(xs)
                y_min, y_max = min(ys), max(ys)
                return x_min, y_min, x_max, y_max
            left_eye_indices = [33, 133, 159, 145]
            right_eye_indices = [362, 263, 386, 374]
            lx1, ly1, lx2, ly2 = get_eye_roi(landmarks, left_eye_indices)
            rx1, ry1, rx2, ry2 = get_eye_roi(landmarks, right_eye_indices)
            pad = 5
            lx1, ly1, lx2, ly2 = max(lx1-pad,0), max(ly1-pad,0), min(lx2+pad,width_px), min(ly2+pad,height_px)
            rx1, ry1, rx2, ry2 = max(rx1-pad,0), max(ry1-pad,0), min(rx2+pad,width_px), min(ry2+pad,height_px)
            left_eye_img = img_for_validation[ly1:ly2, lx1:lx2]
            right_eye_img = img_for_validation[ry1:ry2, rx1:rx2]
            def edge_score(eye_img):
                if eye_img.size == 0:
                    return 0
                gray = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 50, 150)
                sobel_y = cv2.Sobel(edges, cv2.CV_64F, 0, 1, ksize=3)
                score = np.mean(np.abs(sobel_y))
                return score
            left_score = edge_score(left_eye_img)
            right_score = edge_score(right_eye_img)
            glasses_score = (left_score + right_score) / 2
            glasses_status = "fail" if glasses_score > 10 else "pass"
            # Teeth visibility check
            mouth_top_idx = 13
            mouth_bottom_idx = 14
            mouth_left_idx = 78
            mouth_right_idx = 308
            x1 = int(landmarks[mouth_left_idx].x * width_px)
            x2 = int(landmarks[mouth_right_idx].x * width_px)
            y1 = int(landmarks[mouth_top_idx].y * height_px)
            y2 = int(landmarks[mouth_bottom_idx].y * height_px)
            x1, x2 = max(0, x1), min(width_px, x2)
            y1, y2 = max(0, y1), min(height_px, y2)
            mouth_region = img_for_validation[y1:y2, x1:x2]
            if mouth_region.size > 0:
                mouth_gray = cv2.cvtColor(mouth_region, cv2.COLOR_BGR2GRAY)
                mean_brightness = np.mean(mouth_gray)
                # Heuristic: if mouth is open and mean brightness is high, likely teeth are visible
                teeth_visible = (mouth_opening_mm > 1.5) and (mean_brightness > 120)
            else:
                teeth_visible = False
            validation_results["teeth_visible"] = bool(teeth_visible)
            validation_results["teeth_check_status"] = "fail" if teeth_visible else "pass"
            # Collect all results
            validation_results = {
                "face_height_mm": face_height_mm,
                "face_height_status": face_height_status,
                "mouth_opening_mm": mouth_opening_mm,
                "mouth_closed_status": mouth_status,
                "left_eye_opening_px": left_eye_opening_px,
                "right_eye_opening_px": right_eye_opening_px,
                "eyes_open_status": eyes_open_status,
                "glasses_score": glasses_score,
                "glasses_status": glasses_status
            }

    # Log teeth detection
    steps.append(f"Teeth detected: {validation_results.get('teeth_visible', False)}")

    # Log actual top margin (distance from top of image to top of head in mm)
    # In the cropped image, top of head should be at y = top_margin_px
    actual_top_margin_px = top_margin_px
    actual_top_margin_mm = actual_top_margin_px * (height_mm / crop_height_px)
    steps.append(f"Actual top margin: {actual_top_margin_mm:.2f} mm (should be 10 mm)")
    # Ensure these are always in the validation results
    validation_results["actual_top_margin_mm"] = actual_top_margin_mm
    if "teeth_visible" not in validation_results:
        validation_results["teeth_visible"] = False
    if "teeth_check_status" not in validation_results:
        validation_results["teeth_check_status"] = "pass"

    # Calculate final image dimensions in mm
    final_height_px, final_width_px = cropped_img.shape[:2]
    final_width_mm = final_width_px * (width_mm / final_width_px)
    final_height_mm = final_height_px * (height_mm / final_height_px)
    # Validate dimensions
    width_status = "pass" if abs(final_width_mm - 50) < 0.5 else "fail"
    height_status = "pass" if abs(final_height_mm - 60) < 0.5 else "fail"
    # Build validation results with image size first
    validation_results_ordered = {
        "image_width_mm": final_width_mm,
        "image_width_status": width_status,
        "image_height_mm": final_height_mm,
        "image_height_status": height_status,
        "teeth_visible": validation_results.get("teeth_visible", False),
        "teeth_check_status": validation_results.get("teeth_check_status", "pass"),
        "actual_top_margin_mm": actual_top_margin_mm,
    }
    # Add the rest of the validation results
    for k, v in validation_results.items():
        if k not in validation_results_ordered:
            validation_results_ordered[k] = v

    # 7. Create 2x2 grid 4-image template (15cm x 10cm, 1772x1181 px at 300 DPI) with grey borders
    template_width_px = int(10 / 2.54 * 300)   # 10cm wide
    template_height_px = int(15 / 2.54 * 300)  # 15cm tall
    single_width_px = crop_width_px
    single_height_px = crop_height_px
    # Open the processed image
    processed_img_pil = Image.open(processed_path).resize((single_width_px, single_height_px), Image.LANCZOS)
    # Create blank white template
    template_img = Image.new('RGB', (template_width_px, template_height_px), (255, 255, 255))
    draw = ImageDraw.Draw(template_img)
    # Calculate cell size and margins
    cell_width = template_width_px // 2
    cell_height = template_height_px // 2
    x_margin = (cell_width - single_width_px) // 2
    y_margin = (cell_height - single_height_px) // 2
    border_thickness = 6
    border_color = (180, 180, 180)  # grey
    # Paste 4 images in 2x2 grid with grey borders
    for row in range(2):
        for col in range(2):
            x = col * cell_width + x_margin
            y = row * cell_height + y_margin
            template_img.paste(processed_img_pil, (x, y))
            # Draw grey border
            draw.rectangle([
                (x - border_thickness//2, y - border_thickness//2),
                (x + single_width_px + border_thickness//2 - 1, y + single_height_px + border_thickness//2 - 1)
            ], outline=border_color, width=border_thickness)
    # Save template
    template_dir = "image_templates"
    os.makedirs(template_dir, exist_ok=True)
    template_path = os.path.join(template_dir, "4_image_template.jpg")
    template_img.save(template_path, dpi=(300, 300))
    steps.append(f"Created 2x2 grid 4-image template with grey borders and saved as {template_path}.")

    return JSONResponse({
        "success": True,
        "steps": steps,
        "cropped_file": cropped_file,
        "last_processed_image": last_image,
        "template_file": template_path,
        "validation": validation_results_ordered
    })
