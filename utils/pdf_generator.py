import os
import tempfile
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

def generate_pdf_report(report_text):
    """
    Generate a clean PDF from report text.
    Returns absolute path to the generated PDF.
    """
    filename = os.path.join(tempfile.gettempdir(), "deliveryiq_report.pdf")

    c = canvas.Canvas(filename, pagesize=letter)
    width, height = letter
    margin_left = 50
    margin_bottom = 50
    y = height - 50
    line_height = 16

    # Fonts
    FONT_BOLD   = "Helvetica-Bold"
    FONT_NORMAL = "Helvetica"
    FONT_SIZE   = 10

    lines = str(report_text).split("\n")

    for raw_line in lines:
        # Section headers get bold + slightly larger
        is_header = (
            raw_line.startswith("===") or
            raw_line.startswith("---") or
            (len(raw_line) > 2 and raw_line[0].isdigit() and raw_line[1] == ".")
        )

        if is_header:
            c.setFont(FONT_BOLD, FONT_SIZE)
        else:
            c.setFont(FONT_NORMAL, FONT_SIZE)

        # Word-wrap lines that are too long for the page
        max_chars = 95
        if len(raw_line) <= max_chars:
            sub_lines = [raw_line]
        else:
            words = raw_line.split()
            sub_lines = []
            current = ""
            for word in words:
                if len(current) + len(word) + 1 <= max_chars:
                    current = (current + " " + word).strip()
                else:
                    if current:
                        sub_lines.append(current)
                    current = word
            if current:
                sub_lines.append(current)

        for sub in sub_lines:
            if y < margin_bottom + line_height:
                c.showPage()
                c.setFont(FONT_NORMAL, FONT_SIZE)
                y = height - 50

            c.drawString(margin_left, y, sub)
            y -= line_height

    c.save()
    return filename
