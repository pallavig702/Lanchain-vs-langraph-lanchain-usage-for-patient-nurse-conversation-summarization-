from reportlab.graphics.shapes import Drawing, Rect, String, Line
from reportlab.graphics import renderPM, renderSVG
from reportlab.lib import colors

# --- Step 1: Configuration ---
width = 1200
height = 400
drawing = Drawing(width, height)

steps = [
    ("1) PREPROCESSING", "Whisper, diarization,\nrole assignment"),
    ("2) EXTRACTORS", "Problems, Observations,\nEmotions, Interventions, Response"),
    ("3) OMAHA MAPPING", "JSON + RAG → Problems\n& Interventions"),
    ("4) REVIEW", "Align SOAP,\nflag bias/gaps"),
    ("5) SUMMARIZATION", "SOAP + Omaha +\nQA checks"),
]

box_w = 200
box_h = 130
gap = 40
start_x = 40
y = 200

# --- Colors ---
fill_color = colors.HexColor("#e6f0fa")
stroke_color = colors.HexColor("#1e3a5f")
text_color = colors.HexColor("#000000")
highlight_color = colors.HexColor("#333333")
band_color = colors.HexColor("#f4f4f4")
band_edge = colors.HexColor("#cccccc")

# --- Step 2: Draw blocks ---
for i, (title, desc) in enumerate(steps):
    x = start_x + i * (box_w + gap)

    # Box
    drawing.add(Rect(x, y, box_w, box_h, rx=10, ry=10,
                     fillColor=fill_color, strokeColor=stroke_color, strokeWidth=2))

    # Title
    drawing.add(String(x + box_w/2, y + box_h - 30, title,
                       fontName="Helvetica-Bold", fontSize=12,
                       fillColor=stroke_color, textAnchor="middle"))

    # Multiline description
    for j, line in enumerate(desc.split("\n")):
        drawing.add(String(x + box_w/2, y + box_h - 55 - (j * 16), line,
                           fontName="Helvetica", fontSize=10,
                           fillColor=text_color, textAnchor="middle"))

    # Arrows (except after the last block)
    if i < len(steps) - 1:
        arrow_x = x + box_w
        next_x = x + box_w + gap
        arrow_y = y + box_h / 2
        drawing.add(Line(arrow_x + 5, arrow_y, next_x - 5, arrow_y,
                         strokeColor=colors.grey, strokeWidth=2))
        drawing.add(Line(next_x - 10, arrow_y + 5, next_x - 5, arrow_y,
                         strokeColor=colors.grey, strokeWidth=2))
        drawing.add(Line(next_x - 10, arrow_y - 5, next_x - 5, arrow_y,
                         strokeColor=colors.grey, strokeWidth=2))

# --- Step 3: Guardrails and CoT bands ---
drawing.add(Rect(0, 90, width, 30, fillColor=band_color, strokeColor=band_edge))
drawing.add(String(width/2, 105,
                   "GUARDRAILS: evidence-only • label constraints • bias rules • retrieval filters",
                   fontName="Helvetica-Bold", fontSize=10,
                   fillColor=highlight_color, textAnchor="middle"))

drawing.add(Rect(0, 40, width, 30, fillColor=band_color, strokeColor=band_edge))
drawing.add(String(width/2, 55,
                   "CHAIN OF THOUGHT: structured reasoning • JSON-only output",
                   fontName="Helvetica-Bold", fontSize=10,
                   fillColor=highlight_color, textAnchor="middle"))

# --- Title ---
drawing.add(String(width/2, 360,
                   "Agentic Summarization Pipeline → Omaha Mapping",
                   fontName="Helvetica-Bold", fontSize=16,
                   fillColor=text_color, textAnchor="middle"))

# --- Step 4: Export ---
renderSVG.drawToFile(drawing, "agentic_pipeline_omaha_clean.svg")
renderPM.drawToFile(drawing, "agentic_pipeline_omaha_clean.png", fmt="PNG")

print("✅ Saved as 'agentic_pipeline_omaha_clean.svg' and '.png'")
