from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import numpy as np
import cv2
import mediapipe as mp
from PIL import Image, ImageDraw
from rembg import remove, new_session
import io
import os
from datetime import datetime

app = FastAPI()


def resize_and_crop(img: Image.Image, target_width: int, target_height: int) -> Image.Image:
    """
    Resize while keeping aspect ratio, then center-crop to exact target size.
    """
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

    left = (new_width - target_width) // 2
    top = (new_height - target_height) // 2
    right = left + target_width
    bottom = top + target_height

    return img.crop((left, top, right, bottom))


@app.post("/validate_photo")
async def validate_photo(
    image: UploadFile = File(...),
    width_mm: float = Form(...),    # e.g., 50
    height_mm: float = Form(...)    # e.g., 60
):
    steps = []
    dpi = 300  # Working at 300 DPI

    # Precompute target pixel dims from requested mm
    crop_width_px = int(width_mm / 25.4 * dpi)
    crop_height_px = int(height_mm / 25.4 * dpi)

    # Biometric spec pieces (mm) as per your logic
    top_margin_mm = 10
    face_height_target_mm = 34
    # bottom margin is whatever remains to make final height, e.g., 16mm for 50x60 example
    bottom_margin_mm = height_mm - top_margin_mm - face_height_target_mm

    top_margin_px = int(top_margin_mm / 25.4 * dpi)
    face_height_px_target = int(face_height_target_mm / 25.4 * dpi)
    bottom_margin_px = crop_height_px - face_height_px_target - top_margin_px

    # Read uploaded image bytes
    contents = await image.read()
    steps.append("Read uploaded image.")

    # Prepare a single rembg session to reuse
    session = new_session("u2net")

    # 1) Remove background + get mask on original image
    output_image = remove(contents, session=session, only_mask=False, return_mask=True)
    if isinstance(output_image, tuple):
        output_image_bytes, mask_bytes = output_image
    else:
        output_image_bytes = output_image
        mask_bytes = None

    steps.append("Removed background using rembg (u2net) and obtained alpha matte.")

    # Load RGBA with background removed (for alpha channel access if needed)
    img_nobg = Image.open(io.BytesIO(output_image_bytes)).convert("RGBA")
    img_nobg_cv = cv2.cvtColor(np.array(img_nobg), cv2.COLOR_RGBA2BGRA)
    orig_height, orig_width = img_nobg_cv.shape[:2]

    # Alpha matte array
    if mask_bytes is not None:
        alpha_matte = Image.open(io.BytesIO(mask_bytes)).convert("L")
        alpha_np = np.array(alpha_matte)
    else:
        # fallback to alpha from RGBA result
        alpha_np = img_nobg_cv[:, :, 3]

    # Also keep original RGB for face landmarking (MediaPipe prefers natural image)
    img_orig = Image.open(io.BytesIO(contents)).convert("RGB")
    img_orig_cv = cv2.cvtColor(np.array(img_orig), cv2.COLOR_RGB2BGR)

    # 2) Detect face landmarks on the original image
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

        # 3) Find the top of hair using alpha matte at nose tip x-coordinate
        hair_top_y = None
        for y in range(orig_height):
            if alpha_np[y, nose_x] > 10:  # threshold for non-background
                hair_top_y = y
                break
        if hair_top_y is None:
            hair_top_y = 0
        steps.append(f"Detected top of hair at y={hair_top_y} using alpha matte.")

        # 4) Compute crop box: 10mm above hair, 34mm face, remaining bottom margin
        center_x = nose_x
        left = center_x - crop_width_px // 2
        right = left + crop_width_px
        top = hair_top_y - top_margin_px
        bottom = chin_y + bottom_margin_px

        # Pad if out of bounds (white background)
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
            steps.append(
                f"Padded cropped image: top={pad_top}, bottom={pad_bottom}, left={pad_left}, right={pad_right}."
            )

        # Resize to exact target px (e.g., 591x709 for 50x60 mm at 300 DPI)
        cropped_img = cv2.resize(cropped_img, (crop_width_px, crop_height_px), interpolation=cv2.INTER_LANCZOS4)
        steps.append(f"Cropped and resized image to exactly {crop_width_px}x{crop_height_px} px.")

        # Save cropped image (overwrite each run)
        cropped_dir = "cropped_images"
        os.makedirs(cropped_dir, exist_ok=True)
        cropped_path = os.path.join(cropped_dir, "cropped_passport_photo.jpg")
        cv2.imwrite(cropped_path, cropped_img)
        steps.append(f"Saved cropped image as {cropped_path}.")

    # 5) Remove background from the cropped image (and flatten to white)
    with open(cropped_path, "rb") as f:
        cropped_bytes = f.read()

    processed_bytes = remove(cropped_bytes, session=session)
    processed_img = Image.open(io.BytesIO(processed_bytes)).convert("RGBA")

    if processed_img.mode == "RGBA":
        white_bg = Image.new("RGBA", processed_img.size, (255, 255, 255, 255))
        processed_img = Image.alpha_composite(white_bg, processed_img).convert("RGB")
        steps.append("Flattened alpha channel onto white background.")

    processed_dir = "processed_images"
    os.makedirs(processed_dir, exist_ok=True)
    processed_path = os.path.join(processed_dir, "processed_image.jpg")
    processed_img_cv = cv2.cvtColor(np.array(processed_img), cv2.COLOR_RGB2BGR)
    cv2.imwrite(processed_path, processed_img_cv)
    steps.append(f"Saved processed image as {processed_path}.")

    cropped_file = cropped_path
    last_image = processed_path

    # 6) Biometric validation on the cropped image
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

            # Face height (chin to "top of head" landmark index 10 per MediaPipe topology)
            chin_idx = 152
            top_head_idx = 10
            chin_y_v = int(landmarks[chin_idx].y * height_px)
            top_head_y_v = int(landmarks[top_head_idx].y * height_px)
            face_height_px = abs(chin_y_v - top_head_y_v)
            face_height_mm = face_height_px * mm_per_px_h
            face_height_status = "pass" if 30 <= face_height_mm <= 37 else "fail"

            # Mouth closed check
            upper_lip_idx = 13
            lower_lip_idx = 14
            upper_lip_y = int(landmarks[upper_lip_idx].y * height_px)
            lower_lip_y = int(landmarks[lower_lip_idx].y * height_px)
            mouth_opening_px = abs(lower_lip_y - upper_lip_y)
            mouth_opening_mm = mouth_opening_px * mm_per_px_h
            mouth_status = "pass" if mouth_opening_mm < 2.0 else "fail"

            # Eyes open check (simple eyelid distance heuristic)
            left_eye_top_idx = 159
            left_eye_bottom_idx = 145
            right_eye_top_idx = 386
            right_eye_bottom_idx = 374
            left_eye_opening_px = abs(
                int(landmarks[left_eye_bottom_idx].y * height_px) - int(landmarks[left_eye_top_idx].y * height_px)
            )
            right_eye_opening_px = abs(
                int(landmarks[right_eye_bottom_idx].y * height_px) - int(landmarks[right_eye_top_idx].y * height_px)
            )
            left_eye_open = left_eye_opening_px > 5
            right_eye_open = right_eye_opening_px > 5
            eyes_open_status = "pass" if left_eye_open and right_eye_open else "fail"

            # Glasses detection (simple edge score around eye regions)
            def get_eye_roi(lm, indices):
                xs = [int(lm[i].x * width_px) for i in indices]
                ys = [int(lm[i].y * height_px) for i in indices]
                x_min, x_max = min(xs), max(xs)
                y_min, y_max = min(ys), max(ys)
                return x_min, y_min, x_max, y_max

            left_eye_indices = [33, 133, 159, 145]
            right_eye_indices = [362, 263, 386, 374]
            lx1, ly1, lx2, ly2 = get_eye_roi(landmarks, left_eye_indices)
            rx1, ry1, rx2, ry2 = get_eye_roi(landmarks, right_eye_indices)
            pad = 5
            lx1, ly1, lx2, ly2 = max(lx1 - pad, 0), max(ly1 - pad, 0), min(lx2 + pad, width_px), min(ly2 + pad, height_px)
            rx1, ry1, rx2, ry2 = max(rx1 - pad, 0), max(ry1 - pad, 0), min(rx2 + pad, width_px), min(ry2 + pad, height_px)

            left_eye_img = img_for_validation[ly1:ly2, lx1:lx2]
            right_eye_img = img_for_validation[ry1:ry2, rx1:rx2]

            def edge_score(eye_img):
                if eye_img.size == 0:
                    return 0.0
                gray = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 50, 150)
                sobel_y = cv2.Sobel(edges, cv2.CV_64F, 0, 1, ksize=3)
                return float(np.mean(np.abs(sobel_y)))

            left_score = edge_score(left_eye_img)
            right_score = edge_score(right_eye_img)
            glasses_score = (left_score + right_score) / 2.0
            glasses_status = "fail" if glasses_score > 10 else "pass"

            # Teeth visibility (heuristic)
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
            teeth_visible = False
            if x2 > x1 and y2 > y1:
                mouth_region = img_for_validation[y1:y2, x1:x2]
                if mouth_region.size > 0:
                    mouth_gray = cv2.cvtColor(mouth_region, cv2.COLOR_BGR2GRAY)
                    mean_brightness = float(np.mean(mouth_gray))
                    teeth_visible = (mouth_opening_mm > 1.5) and (mean_brightness > 120)

            # Collect measured results
            validation_results = {
                "face_height_mm": float(face_height_mm),
                "face_height_status": face_height_status,
                "mouth_opening_mm": float(mouth_opening_mm),
                "mouth_closed_status": mouth_status,
                "left_eye_opening_px": int(left_eye_opening_px),
                "right_eye_opening_px": int(right_eye_opening_px),
                "eyes_open_status": eyes_open_status,
                "glasses_score": float(glasses_score),
                "glasses_status": glasses_status,
                "teeth_visible": bool(teeth_visible),
                "teeth_check_status": "fail" if teeth_visible else "pass",
            }

    # 7) Log and compute margins/dimensions in mm
    steps.append(f"Teeth detected: {validation_results.get('teeth_visible', False)}")

    # In the cropped image, top of head should be at y = top_margin_px by construction
    actual_top_margin_mm = top_margin_px * (height_mm / crop_height_px)
    steps.append(f"Actual top margin: {actual_top_margin_mm:.2f} mm (should be {top_margin_mm} mm)")
    validation_results["actual_top_margin_mm"] = float(actual_top_margin_mm)

    # Final image dimensions validation (based on requested width_mm/height_mm target)
    final_height_px, final_width_px = cropped_img.shape[:2]
    final_width_mm = final_width_px * (width_mm / final_width_px)
    final_height_mm = final_height_px * (height_mm / final_height_px)

    width_status = "pass" if abs(final_width_mm - width_mm) < 0.5 else "fail"
    height_status = "pass" if abs(final_height_mm - height_mm) < 0.5 else "fail"

    # Ordered output with image size first
    validation_results_ordered = {
        "image_width_mm": float(final_width_mm),
        "image_width_status": width_status,
        "image_height_mm": float(final_height_mm),
        "image_height_status": height_status,
        "teeth_visible": bool(validation_results.get("teeth_visible", False)),
        "teeth_check_status": validation_results.get("teeth_check_status", "pass"),
        "actual_top_margin_mm": float(actual_top_margin_mm),
    }
    for k, v in validation_results.items():
        if k not in validation_results_ordered:
            validation_results_ordered[k] = v

    # 8) Create 2x2 grid template (10cm x 15cm = 1181 x 1772 px at 300 DPI) with grey borders
    template_width_px = int(10 / 2.54 * 300)   # width in px
    template_height_px = int(15 / 2.54 * 300)  # height in px

    single_width_px = crop_width_px
    single_height_px = crop_height_px

    processed_img_pil = Image.open(processed_path).resize((single_width_px, single_height_px), Image.LANCZOS)

    template_img = Image.new('RGB', (template_width_px, template_height_px), (255, 255, 255))
    draw = ImageDraw.Draw(template_img)

    cell_width = template_width_px // 2
    cell_height = template_height_px // 2
    x_margin = (cell_width - single_width_px) // 2
    y_margin = (cell_height - single_height_px) // 2

    border_thickness = 6
    border_color = (180, 180, 180)

    for row in range(2):
        for col in range(2):
            x = col * cell_width + x_margin
            y = row * cell_height + y_margin
            template_img.paste(processed_img_pil, (x, y))
            draw.rectangle(
                [
                    (x - border_thickness // 2, y - border_thickness // 2),
                    (x + single_width_px + border_thickness // 2 - 1,
                     y + single_height_px + border_thickness // 2 - 1),
                ],
                outline=border_color,
                width=border_thickness,
            )

    template_dir = "image_templates"
    os.makedirs(template_dir, exist_ok=True)
    template_path = os.path.join(template_dir, "4_image_template.jpg")
    template_img.save(template_path, dpi=(300, 300))
    steps.append(f"Created 2x2 grid template and saved as {template_path}.")

    return JSONResponse({
        "success": True,
        "steps": steps,
        "cropped_file": cropped_file,
        "last_processed_image": last_image,
        "template_file": template_path,
        "validation": validation_results_ordered
    })


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, log_level="info")
