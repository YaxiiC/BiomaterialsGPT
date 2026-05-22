import re
import json
import uuid
from docx import Document

def parse_docx_to_mcq(docx_path):
    doc = Document(docx_path)
    mcqs = []
    current_section = None
    current_mcq = None
    q_counter = 0

    # regexes
    opt_re  = re.compile(r'^([a-dA-D])[)\.]?\s*(.+)')
    ans_re  = re.compile(r'^Answer[:\s]+([a-dA-D])', re.I)
    expl_re = re.compile(r'^Explanation[:\s]+(.+)', re.I)

    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue

        # 1) Section header: no newline, no 'Answer'/'Explanation' and no inline options
        if '\n' not in text and not ans_re.match(text) and not expl_re.match(text):
            current_section = text
            continue

        # 2) Question + options in one paragraph (detected by '\na)')
        if '\na)' in text:
            q_counter += 1
            parts = text.split('\n')
            question_text = parts[0].strip()
            options = {}
            for line in parts[1:]:
                m = opt_re.match(line.strip())
                if m:
                    letter, opttext = m.groups()
                    options[letter.lower()] = opttext.strip()

            current_mcq = {
                "id":          str(uuid.uuid4()),
                "section":     current_section or "",
                "number":      str(q_counter),
                "question":    question_text,
                "options":     options,
                "answer":      None,
                "explanation": None
            }
            mcqs.append(current_mcq)
            continue

        # 3) Answer + Explanation paragraph
        if text.startswith("Answer"):
            parts = text.split('\n')
            for line in parts:
                m = ans_re.match(line.strip())
                if m and current_mcq:
                    current_mcq["answer"] = m.group(1).lower()
                m2 = expl_re.match(line.strip())
                if m2 and current_mcq:
                    current_mcq["explanation"] = m2.group(1).strip()
            continue

        # 4) (Rare) continuation of explanation
        if current_mcq and current_mcq.get("explanation") is not None:
            current_mcq["explanation"] += " " + text
            continue

    return mcqs



from paths import REPO_ROOT, SCQ_BANK_JSON

if __name__ == "__main__":
    input_docx = REPO_ROOT / "local" / "single_choice_round1_readyforreview.docx"
    output_json = SCQ_BANK_JSON

    mcq_list = parse_docx_to_mcq(str(input_docx))
    print(f"Parsed {len(mcq_list)} questions.")

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(mcq_list, f, ensure_ascii=False, indent=2)

    print(f"Saved SCQ bank to '{output_json}'")