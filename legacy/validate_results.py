
import json
import re
from dataclasses import dataclass, field, asdict

# --- Extracted logic from pramana_test.py ---

YAML_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)
HEADER_RE = re.compile(r"^##\s+(.+?)$", re.MULTILINE)

REQUIRED_HEADERS_PREFIXES = [
    "samshaya",
    "pramana",
    "pancha avayava",
    "tarka",
    "hetvabhasa",
    "nirnaya",
]
REQUIRED_PRAMANA_SUBTYPES = ["pratyaksha", "anumana", "upamana", "shabda"]
REQUIRED_AVAYAVA_PARTS = ["pratijna", "hetu", "udaharana", "upanaya", "nigamana"]
REQUIRED_HETVABHASA_TYPES = [
    "savyabhichara", "viruddha", "prakaranasama", "sadhyasama", "kalaatita",
]
REQUIRED_NIRNAYA_FIELDS = ["answer", "confidence", "epistemic status"]

@dataclass
class FormatReport:
    yaml_present: bool = False
    yaml_has_ground_truth: bool = False
    headers_found: list[str] = field(default_factory=list)
    missing_headers: list[str] = field(default_factory=list)
    headers_in_order: bool = False
    pramana_complete: bool = False
    avayava_complete: bool = False
    udaharana_uses_template: bool = False
    hetvabhasa_complete: bool = False
    nirnaya_complete: bool = False
    placeholders_clean: bool = False

    @property
    def passes(self) -> bool:
        return (
            self.yaml_present
            and self.yaml_has_ground_truth
            and not self.missing_headers
            and self.headers_in_order
            and self.pramana_complete
            and self.avayava_complete
            and self.udaharana_uses_template
            and self.hetvabhasa_complete
            and self.nirnaya_complete
            and self.placeholders_clean
        )

_PRAMANA_NONE_MARKER = "(none in this problem)"
_HETVABHASA_ABSENT_MARKER = "absent"
_MIN_CONTENT_TOKENS = 1

def _bullet_value(section_text: str, label: str) -> str | None:
    pattern = re.compile(
        rf"^\s*[-*]\s*{re.escape(label)}\s*:\s*(.*?)\s*$",
        re.IGNORECASE | re.MULTILINE,
    )
    m = pattern.search(section_text)
    if not m:
        return None
    return m.group(1)

def _is_placeholder_value(value: str, allowed_sentinels: tuple[str, ...]) -> bool:
    stripped = value.strip().strip(".").strip().lower()
    return any(stripped == s.lower() for s in allowed_sentinels)

def _has_real_content(value: str | None,
                      allowed_sentinels: tuple[str, ...] = ()) -> bool:
    if value is None:
        return False
    if _is_placeholder_value(value, allowed_sentinels):
        return True
    cleaned = re.sub(r"<[^>]*>", " ", value)
    tokens = [t for t in re.split(r"\s+", cleaned) if t]
    real_tokens = [
        t for t in tokens
        if not re.fullmatch(r"x+", t, re.IGNORECASE)
        and not re.fullmatch(r"[^\w]+", t)
    ]
    return len(real_tokens) >= _MIN_CONTENT_TOKENS

def _section_text(text: str, header_prefix: str) -> str:
    pattern = re.compile(
        rf"^##\s+{re.escape(header_prefix)}\b.*?$(.*?)(?=^##\s|\Z)",
        re.DOTALL | re.MULTILINE | re.IGNORECASE,
    )
    m = pattern.search(text)
    return m.group(1) if m else ""

def score_format(text: str) -> FormatReport:
    report = FormatReport()
    yaml_match = YAML_RE.search(text)
    if yaml_match:
        report.yaml_present = True
        report.yaml_has_ground_truth = "ground_truth:" in yaml_match.group(1).lower()

    headers = [h.strip().lower() for h in HEADER_RE.findall(text)]
    report.headers_found = headers
    for prefix in REQUIRED_HEADERS_PREFIXES:
        if not any(h.startswith(prefix) for h in headers):
            report.missing_headers.append(prefix)

    order_indices: list[int] = []
    for h in headers:
        for idx, prefix in enumerate(REQUIRED_HEADERS_PREFIXES):
            if h.startswith(prefix):
                order_indices.append(idx)
                break
    report.headers_in_order = (
        not report.missing_headers
        and order_indices == sorted(order_indices)
        and len(set(order_indices)) == len(REQUIRED_HEADERS_PREFIXES)
    )

    pramana_section = _section_text(text, "pramana")
    pramana_lower = pramana_section.lower()
    report.pramana_complete = all(
        s in pramana_lower for s in REQUIRED_PRAMANA_SUBTYPES
    )

    avayava_section = _section_text(text, "pancha avayava")
    avayava_lower = avayava_section.lower()
    report.avayava_complete = all(p in avayava_lower for p in REQUIRED_AVAYAVA_PARTS)
    report.udaharana_uses_template = (
        "wherever" in avayava_lower and "there is" in avayava_lower
    )

    hetvabhasa_section = _section_text(text, "hetvabhasa")
    hetvabhasa_lower = hetvabhasa_section.lower()
    report.hetvabhasa_complete = all(
        h in hetvabhasa_lower for h in REQUIRED_HETVABHASA_TYPES
    )

    nirnaya_section = _section_text(text, "nirnaya")
    nirnaya_lower = nirnaya_section.lower()
    report.nirnaya_complete = all(
        f in nirnaya_lower for f in REQUIRED_NIRNAYA_FIELDS
    )

    placeholders_clean = True
    for label in REQUIRED_PRAMANA_SUBTYPES:
        if not _has_real_content(
            _bullet_value(pramana_section, label),
            allowed_sentinels=(_PRAMANA_NONE_MARKER,),
        ):
            placeholders_clean = False
            break
    if placeholders_clean:
        for label in REQUIRED_AVAYAVA_PARTS:
            if not _has_real_content(_bullet_value(avayava_section, label)):
                placeholders_clean = False
                break
    if placeholders_clean:
        for label in REQUIRED_HETVABHASA_TYPES:
            if not _has_real_content(
                _bullet_value(hetvabhasa_section, label),
                allowed_sentinels=(_HETVABHASA_ABSENT_MARKER,),
            ):
                placeholders_clean = False
                break
    report.placeholders_clean = placeholders_clean
    return report

_ONES = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
    "seventeen", "eighteen", "nineteen",
]
_TENS = [
    "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
]

def _int_to_word(n: int) -> str:
    if 0 <= n < 20:
        return _ONES[n]
    if 20 <= n < 100:
        tens, ones = divmod(n, 10)
        return _TENS[tens] if ones == 0 else f"{_TENS[tens]}-{_ONES[ones]}"
    if n == 100:
        return "one hundred"
    return str(n)

_DIGIT_TO_WORD: dict[str, str] = {str(n): _int_to_word(n) for n in range(0, 101)}
_WORD_TO_DIGIT: dict[str, str] = {v: k for k, v in _DIGIT_TO_WORD.items()}

def _expand_number_synonyms(token: str) -> list[str]:
    out = {token}
    if token.isdigit() and token in _DIGIT_TO_WORD:
        out.add(_DIGIT_TO_WORD[token])
    if token in _WORD_TO_DIGIT:
        out.add(_WORD_TO_DIGIT[token])
    def _sub_digits(match: re.Match[str]) -> str:
        d = match.group(0)
        return _DIGIT_TO_WORD.get(d, d)
    swapped = re.sub(r"\b\d{1,3}\b", _sub_digits, token)
    if swapped != token:
        out.add(swapped)
    for word, digit in _WORD_TO_DIGIT.items():
        if word and re.search(rf"\b{re.escape(word)}\b", token):
            out.add(re.sub(rf"\b{re.escape(word)}\b", digit, token))
    return [s.lower() for s in out]

def _group_matches(haystack: str, group: list[str]) -> bool:
    for raw in group:
        for variant in _expand_number_synonyms(raw.lower()):
            if not variant:
                continue
            if re.search(rf"\b{re.escape(variant)}\b", haystack):
                return True
    return False

@dataclass
class Problem:
    id: str
    question: str
    ground_truth_keywords: list[list[str]]
    source: str

def score_semantic(text: str, problem: Problem) -> bool:
    if not problem.ground_truth_keywords:
        return False
    nirnaya = _section_text(text, "nirnaya").lower()
    if not nirnaya:
        yaml_match = YAML_RE.search(text)
        nirnaya = (yaml_match.group(1) if yaml_match else "").lower()
    if not nirnaya:
        return False
    return all(_group_matches(nirnaya, group) for group in problem.ground_truth_keywords)

# --- End of extracted logic ---

def main():
    with open('pramana_test_results.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    results = data['results']
    model_name = data['model']
    discrepancies = []

    for item in results:
        test_id = item['id']
        raw_response = item['response']
        recorded_format_pass = item['format_pass']
        recorded_semantic_pass = item['semantic_pass']
        
        # Re-score
        problem = Problem(
            id=item['id'],
            question=item['question'],
            ground_truth_keywords=item['ground_truth_keywords'],
            source=item['source']
        )
        
        new_format_report = score_format(raw_response)
        new_format_pass = new_format_report.passes
        new_semantic_pass = score_semantic(raw_response, problem)
        
        # Check all fields in format_report
        recorded_report = item['format_report']
        calculated_report = asdict(new_format_report)
        
        report_mismatch = False
        for key in calculated_report:
            if calculated_report[key] != recorded_report.get(key):
                report_mismatch = True
                break

        if report_mismatch:
            discrepancies.append({
                'id': test_id,
                'model': model_name,
                'type': 'format_report_detail',
                'recorded': recorded_report,
                'calculated': calculated_report
            })
        
        if new_format_pass != recorded_format_pass:
            discrepancies.append({
                'id': test_id,
                'model': model_name,
                'type': 'format_pass_boolean',
                'recorded': recorded_format_pass,
                'calculated': new_format_pass
            })
            
        if new_semantic_pass != recorded_semantic_pass:
            discrepancies.append({
                'id': test_id,
                'model': model_name,
                'type': 'semantic',
                'recorded': recorded_semantic_pass,
                'calculated': new_semantic_pass
            })

    if not discrepancies:
        print("SUCCESS: All scores match perfectly.")
    else:
        print(f"FOUND {len(discrepancies)} DISCREPANCIES:")
        for d in discrepancies:
            print(f"Test ID: {d['id']}, Model: {d['model']}")
            print(f"  Type: {d['type']}")
            print(f"  Recorded: {d['recorded']} vs Calculated: {d['calculated']}")
            if d['type'] == 'format':
                print(f"  Calculated Report: {d['report']}")
                print(f"  Recorded Report: {d['recorded_report']}")
            print("-" * 20)

if __name__ == '__main__':
    main()
