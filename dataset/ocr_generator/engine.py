import os
import random
import numpy as np
import unicodedata2 as unicodedata
from PIL import Image
from fontTools.ttLib import TTFont
import freetype
import uharfbuzz as hb
import regex
import cv2

from config import (
    FONT_SIZE,
    IMAGE_WIDTH,
    BASE_HEIGHT,
    PADDING,
)


# -------------------------------------------------
# FONT LOADING
# -------------------------------------------------

def load_fonts(folder):
    fonts = []
    for root, _, files in os.walk(folder):
        for file in files:
            if file.lower().endswith((".ttf", ".otf")):
                path = os.path.join(root, file)
                try:
                    supported = get_supported_chars(path)
                    fonts.append({
                        "path": path,
                        "supported": supported
                    })
                except:
                    pass
    return fonts


def get_supported_chars(font_path):
    font = TTFont(font_path)
    cmap = font["cmap"].getBestCmap()
    return set(chr(c) for c in cmap.keys())


def coverage_score(sentence, supported_chars):
    score = 0
    for c in sentence:
        if c in supported_chars:
            score += 1
        elif unicodedata.category(c) in ("Zs", "Cf", "Po"):
            # allow spaces and punctuation
            score += 1
    return score


def font_can_render(sentence, font_path):
    """
    Ensure shaping works and glyphs exist in FreeType.
    """

    try:
        face, infos, positions, _ = shape_text(sentence, font_path)

        if face is None or not infos:
            return False

        for info in infos:
            glyph_id = info.codepoint

            if glyph_id >= face.num_glyphs:
                return False

            try:
                face.load_glyph(glyph_id)
            except:
                return False

        return True

    except Exception:
        return False


def choose_best_font(sentence, fonts):
    if not sentence:
        return None

    best_fonts = []
    max_score = 0

    # ---- Pass 1: Unicode coverage
    for font in fonts:

        score = coverage_score(sentence, font["supported"])

        if score > max_score:
            max_score = score
            best_fonts = [font]

        elif score == max_score:
            best_fonts.append(font)

    if not best_fonts:
        return None

    # ---- Pass 2: HarfBuzz + FreeType validation
    valid_fonts = []

    for font in best_fonts:
        if font_can_render(sentence, font["path"]):
            valid_fonts.append(font)

    if not valid_fonts:
        return None

    return random.choice(valid_fonts)


# -------------------------------------------------
# DIRECTION FILTERING
# -------------------------------------------------


def dominant_direction(text):
    rtl = 0
    ltr = 0

    for c in text:
        bidi = unicodedata.bidirectional(c)

        # Ignore formatting controls in counting
        if unicodedata.category(c) == "Cf":
            continue

        if bidi in ('R', 'AL', 'AN'):
            rtl += 1
        elif bidi in ('L', 'EN'):
            ltr += 1

    return 'rtl' if rtl > ltr else 'ltr'


def filter_by_dominant_direction(text):
    direction = dominant_direction(text)
    filtered = []

    for c in text:
        bidi = unicodedata.bidirectional(c)
        category = unicodedata.category(c)

        # ✅ Always preserve format controls (RLO, RLE, etc.)
        if category == "Cf":
            filtered.append(c)
            continue

        if bidi == '':
            filtered.append(c)

        if direction == 'rtl':
            if bidi in ('R', 'AL', 'AN'):
                filtered.append(c)
            elif bidi.startswith('N') or bidi in ('WS', 'CS', 'ES', 'ET'):
                filtered.append(c)
        else:
            if bidi in ('L', 'EN'):
                filtered.append(c)
            elif bidi.startswith('N') or bidi in ('WS', 'CS', 'ES', 'ET'):
                filtered.append(c)

    return "".join(filtered)


# -------------------------------------------------
# SHAPING
# -------------------------------------------------

def shape_text(text, font_path):
    if not text:
        return None, [], [], None

    face = freetype.Face(font_path)
    face.set_pixel_sizes(0, FONT_SIZE)

    with open(font_path, "rb") as f:
        fontdata = f.read()

    hb_face = hb.Face(fontdata)
    hb_font = hb.Font(hb_face)
    hb.ot_font_set_funcs(hb_font)

    hb_font.scale = (
        face.size.x_ppem << 6,
        face.size.y_ppem << 6
    )

    buf = hb.Buffer()
    buf.add_str(text)
    buf.guess_segment_properties()
    hb.shape(hb_font, buf)

    infos = buf.glyph_infos
    positions = buf.glyph_positions

    return face, infos, positions, buf.direction


def measure_text_width(text, font_path):
    _, _, positions, _ = shape_text(text, font_path)
    if not positions:
        return 0
    return sum(pos.x_advance for pos in positions) >> 6


# -------------------------------------------------
# WRAP TEXT
# -------------------------------------------------

def wrap_text(sentence, font_path):
    max_width = IMAGE_WIDTH - 2 * PADDING
    clusters = regex.findall(r'\X', sentence)

    lines = []
    current = ""
    last_space_index = -1

    for cluster in clusters:

        test = current + cluster
        width = measure_text_width(test, font_path)

        if width <= max_width:
            current = test
            if cluster.isspace():
                last_space_index = len(current)
        else:
            if not current:
                lines.append(cluster)
                current = ""
                last_space_index = -1
                continue

            if last_space_index > 0:
                line = current[:last_space_index].rstrip()
                remainder = current[last_space_index:].lstrip()
                lines.append(line)
                current = remainder + cluster
            else:
                lines.append(current)
                current = cluster

            last_space_index = -1

    if current:
        lines.append(current.rstrip())

    return lines


# -------------------------------------------------
# RENDER MULTI-LINE (SAFETY-CORRECT)
# -------------------------------------------------

def render_sentence(sentence, fonts):
    font_data = choose_best_font(sentence, fonts)

    if not font_data:
        return None, None, None, None

    sentence = "".join(
        c for c in sentence
        if (
                c in font_data["supported"]
                or unicodedata.category(c) in ("Cf", "Zs")
        )
    )

    sentence = filter_by_dominant_direction(sentence)

    if not sentence:
        return None, None, None, None

    font_path = font_data["path"]
    lines = wrap_text(sentence, font_path)

    face = freetype.Face(font_path)
    face.set_char_size(FONT_SIZE * 64)
    face.set_pixel_sizes(0, FONT_SIZE)

    return sentence, face, lines, font_path


def render_face(face, lines, font_path, profile):
    ascent = face.size.ascender >> 6
    descent = abs(face.size.descender >> 6)
    line_height = ascent + descent

    # -------- Height Calculation --------
    max_curve_amp = int(
        abs(profile["curve_range"][1] * (IMAGE_WIDTH / 2) ** 2)
    )

    total_height = (
            len(lines) * line_height +
            2 * PADDING +
            2 * max_curve_amp +
            20
    )

    layout_height = max(BASE_HEIGHT, total_height)
    height = int(layout_height *
                 profile["height_safety_factor"])

    # -------- Width Calculation --------
    max_line_width = 0
    for line in lines:
        _, _, positions, _ = shape_text(line, font_path)
        if positions:
            line_width = sum(pos.x_advance for pos in positions) >> 6
            max_line_width = max(max_line_width, line_width)

    layout_width = max(max_line_width + 2 * PADDING, IMAGE_WIDTH)
    canvas_width = int(layout_width *
                       profile["width_safety_factor"])

    horizontal_offset = (canvas_width - layout_width) // 2

    width = canvas_width

    canvas = np.ones((height, width), dtype=np.uint8) * 255

    y_cursor = PADDING + ascent + max_curve_amp
    curve_strength = random.uniform(*profile["curve_range"])

    for line in lines:

        face, infos, positions, line_direction = shape_text(line, font_path)

        line_width = sum(pos.x_advance for pos in positions) >> 6

        if line_direction == 'rtl':
            pen_x = (
                    horizontal_offset +
                    layout_width - PADDING - line_width
            )
        else:
            pen_x = PADDING + horizontal_offset

        pen_y = y_cursor + random.randint(
            -profile["line_jitter"],
            profile["line_jitter"]
        )

        for info, pos in zip(infos, positions):
            face.load_glyph(info.codepoint, freetype.FT_LOAD_RENDER)

            bitmap = face.glyph.bitmap
            w, h = bitmap.width, bitmap.rows

            x = pen_x + (pos.x_offset >> 6) + face.glyph.bitmap_left
            y = pen_y - (pos.y_offset >> 6) - face.glyph.bitmap_top
            y += int(curve_strength * (pen_x - width / 2) ** 2)

            if w > 0 and h > 0:
                glyph = np.array(bitmap.buffer,
                                 dtype=np.uint8).reshape(h, w)

                if random.random() < profile["stroke_dilate_prob"]:
                    glyph = cv2.dilate(glyph,
                                       np.ones((2, 2), np.uint8), 1)

                if random.random() < profile["stroke_erode_prob"]:
                    glyph = cv2.erode(glyph,
                                      np.ones((2, 2), np.uint8), 1)

                x1, y1 = max(0, x), max(0, y)
                x2, y2 = min(width, x + w), min(height, y + h)

                if x1 < x2 and y1 < y2:
                    canvas[y1:y2, x1:x2] = np.minimum(
                        canvas[y1:y2, x1:x2],
                        255 - glyph[0:(y2 - y1),
                              0:(x2 - x1)]
                    )

            advance = (pos.x_advance >> 6) + random.randint(
                *profile["char_spacing_range"]
            )
            pen_x += advance

        y_cursor += line_height

    return Image.fromarray(canvas).convert("RGB")
