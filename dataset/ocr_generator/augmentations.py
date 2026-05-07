import random
import numpy as np
import cv2
from PIL import Image

def apply_augmentations(img, profile):

    arr = np.array(img).astype(np.uint8)
    h, w = arr.shape[:2]

    # =================================================
    # 1️⃣ BACKGROUND BLENDING
    # =================================================
    if profile["background_path"] is not None:

        bg = cv2.imread(profile["background_path"])

        if bg.shape[0] < h:
            repeat = int(np.ceil(h / bg.shape[0]))
            bg = np.vstack([bg] * repeat)

        if bg.shape[1] < w:
            repeat = int(np.ceil(w / bg.shape[1]))
            bg = np.hstack([bg] * repeat)

        bg_h, bg_w = bg.shape[:2]
        x = random.randint(0, bg_w - w)
        y = random.randint(0, bg_h - h)
        bg_crop = bg[y:y+h, x:x+w]

        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        mask = 1 - (gray.astype(np.float32) / 255.0)
        mask = np.stack([mask]*3, axis=2)

        arr = (bg_crop.astype(np.float32) * (1 - mask)).astype(np.uint8)

        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)

    # =================================================
    # 2️⃣ ROTATION
    # =================================================
    if profile["rotation"] != (0, 0):
        angle = random.uniform(*profile["rotation"])
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        arr = cv2.warpAffine(arr, M, (w, h),
                             borderMode=cv2.BORDER_REPLICATE)

    # =================================================
    # 3️⃣ ELASTIC
    # =================================================
    if profile["elastic"]:
        dx = cv2.GaussianBlur(
            (np.random.rand(h, w).astype(np.float32)*2 - 1),
            (17,17), 0
        ) * 8

        dy = cv2.GaussianBlur(
            (np.random.rand(h, w).astype(np.float32)*2 - 1),
            (17,17), 0
        ) * 8

        x_grid, y_grid = np.meshgrid(np.arange(w),
                                     np.arange(h))

        map_x = (x_grid + dx).astype(np.float32)
        map_y = (y_grid + dy).astype(np.float32)

        arr = cv2.remap(arr, map_x, map_y,
                        interpolation=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REFLECT)

    # =================================================
    # 4️⃣ DROPOUT
    # =================================================
    if profile["dropout"]:
        for _ in range(random.randint(10, 30)):
            x1 = random.randint(0, w-10)
            y1 = random.randint(0, h-10)
            x2 = x1 + random.randint(5, 40)
            y2 = y1 + random.randint(5, 15)
            arr[y1:y2, x1:x2] = 255

    # =================================================
    # 5️⃣ MOTION BLUR
    # =================================================
    if profile["motion_blur"]:
        k = random.choice([3,5,7])
        kernel = np.zeros((k,k))
        kernel[k//2,:] = 1
        kernel /= k
        arr = cv2.filter2D(arr,-1,kernel)

    # =================================================
    # 6️⃣ NOISE
    # =================================================
    if profile["noise"]:
        noise = np.random.normal(0, 8, arr.shape).astype(np.float32)
        arr = np.clip(arr.astype(np.float32) + noise,
                      0, 255).astype(np.uint8)

    # =================================================
    # 7️⃣ LOW RESOLUTION
    # =================================================
    if profile["low_res"]:
        scale = random.uniform(0.4, 0.7)
        small = cv2.resize(arr, None,
                           fx=scale, fy=scale,
                           interpolation=cv2.INTER_AREA)
        arr = cv2.resize(small, (w, h),
                         interpolation=cv2.INTER_LINEAR)

    # =================================================
    # 8️⃣ INK FADE
    # =================================================
    if profile["ink_fade"]:
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY).astype(np.float32)

        fade_strength = random.uniform(0.5, 0.85)
        faded = gray * fade_strength + 255 * (1 - fade_strength)

        texture_noise = np.random.normal(0, 10, gray.shape)
        faded = np.clip(faded + texture_noise,
                        0, 255).astype(np.uint8)

        arr = cv2.cvtColor(faded, cv2.COLOR_GRAY2RGB)

    # =================================================
    # 9️⃣ JPEG
    # =================================================
    if profile["jpeg"]:
        quality = random.randint(30, 80)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encimg = cv2.imencode('.jpg', arr, encode_param)
        arr = cv2.imdecode(encimg, 1)

    # =================================================
    # 🔟 PERSPECTIVE
    # =================================================
    if profile["perspective"]:
        margin = int(min(w,h)*0.1)

        pts1 = np.float32([[0,0],[w,0],[0,h],[w,h]])
        pts2 = np.float32([
            [random.randint(0,margin),
             random.randint(0,margin)],
            [w-random.randint(0,margin),
             random.randint(0,margin)],
            [random.randint(0,margin),
             h-random.randint(0,margin)],
            [w-random.randint(0,margin),
             h-random.randint(0,margin)]
        ])

        M = cv2.getPerspectiveTransform(pts1,pts2)
        arr = cv2.warpPerspective(arr,M,(w,h),
                                  borderMode=cv2.BORDER_REPLICATE)

    return Image.fromarray(arr.astype(np.uint8))