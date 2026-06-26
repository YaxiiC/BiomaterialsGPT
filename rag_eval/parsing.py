import re


def extract_letter(response, options):
    m = re.search(r"ANSWER:\s*([a-dA-DtTfF])", response)
    if m:
        return m.group(1).lower()
    for line in reversed(response.strip().split("\n")):
        for key in options:
            low = line.strip().lower()
            if low.startswith(key + ")") or low == key:
                return key.lower()
    return "?"


def extract_source(response):
    m = re.search(r"SOURCE:\s*(CONTEXT|OWN)", response, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    return "UNKNOWN"


def strip_tags(response):
    return re.sub(r"\n?SOURCE:\s*(CONTEXT|OWN)\s*$", "", response,
                  flags=re.IGNORECASE).strip()
