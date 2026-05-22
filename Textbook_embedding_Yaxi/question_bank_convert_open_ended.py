import re
import json
import uuid
from docx import Document

def is_numbered_list(paragraph):
    """
    Returns True if this paragraph is part of a Word numbered list
    (i.e. it has a <w:numPr> element in its XML).
    """
    pPr = paragraph._p.pPr
    return pPr is not None and getattr(pPr, 'numPr', None) is not None

def parse_docx_to_question_bank(docx_path):
    doc = Document(docx_path)
    questions = []
    current_unit = None
    current_part = None
    list_counter = 0

    # regexes for explicit headings / inline numbering
    unit_re     = re.compile(r'^UNIT[-\s–]*(\d+)', re.IGNORECASE)
    part_re     = re.compile(r'^PART[-\s–]*([AB])', re.IGNORECASE)
    inline_q_re = re.compile(r'^(\d+)[\.\)]\s*(.+)')

    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue

        # 1) UNIT header?
        m = unit_re.match(text)
        if m:
            current_unit = f"UNIT-{m.group(1)}"
            continue

        # 2) PART header?
        m = part_re.match(text)
        if m:
            current_part = f"PART-{m.group(1).upper()}"
            list_counter = 0
            continue

        # 3) Inline-numbered question (“1. …”)
        m = inline_q_re.match(text)
        if m:
            num, qtext = m.groups()
            list_counter = int(num)
            questions.append({
                "id":       str(uuid.uuid4()),
                "unit":     current_unit or "",
                "part":     current_part or "",
                "number":   num,
                "question": qtext
            })
            continue

        # 4) Word’s auto-numbered list paragraph
        if is_numbered_list(para):
            list_counter += 1
            questions.append({
                "id":       str(uuid.uuid4()),
                "unit":     current_unit or "",
                "part":     current_part or "",
                "number":   str(list_counter),
                "question": text
            })
            continue

        # 5) Continuation of the last question
        if questions:
            questions[-1]["question"] += " " + text

    return questions

from paths import OPEN_ENDED_BANK_JSON, REPO_ROOT

if __name__ == "__main__":
    input_docx = REPO_ROOT / "local" / "open-ended_Q&A_round1_all.docx"
    output_json = OPEN_ENDED_BANK_JSON

    qb = parse_docx_to_question_bank(str(input_docx))
    print(f"Parsed {len(qb)} questions from {input_docx}")

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(qb, f, ensure_ascii=False, indent=2)

    print(f"Saved question bank to '{output_json}'")



