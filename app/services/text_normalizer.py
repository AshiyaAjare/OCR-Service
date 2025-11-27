# app/services/text_normalizer.py

import re
from typing import List


_whitespace_re = re.compile(r"\s+")


def normalize_text_to_lines(text: str) -> List[str]:
    """
    Convert a big multi-line string into a list of clean lines:
    - split on actual line breaks
    - strip leading/trailing spaces
    - collapse multiple spaces/tabs inside line
    - drop empty lines
    """
    lines: List[str] = []
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        # Collapse all whitespace inside to a single space
        cleaned = _whitespace_re.sub(" ", stripped)
        lines.append(cleaned)
    return lines
