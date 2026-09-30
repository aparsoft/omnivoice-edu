"""Text preprocessor for TTS.

Turns raw text — the kind produced by LLMs, textbooks and web pages — into
plain, speakable sentences before it reaches the TTS model. Speech models
read symbols, markup and digits unreliably; this pipeline rewrites them as
the words a teacher would say aloud.

Handles:
- Math operators and symbols (=, +, ×, ÷, <, >, %, etc.) → speakable words
- LaTeX notation (fractions, roots, sums, integrals, Greek letters)
- Number verbalization (standalone numbers → English words)
- Currency verbalization ($, £, €, ¥, ₹), including lakh/crore scales
- Dates, times, temperatures, ordinals and phone numbers
- Markdown/HTML flattening (code blocks, links, headings, bold, italic)
- URL and email removal
- Unicode special characters (arrows, dashes, bullets, quotes, fractions,
  superscripts, subscripts)
- Emoji removal and punctuation normalization
- Invisible Unicode removal (ZWNJ/ZWJ are kept for Devanagari)

OmniVoice's inline control tags — non-verbal sounds such as ``[laughter]``
and CMU pronunciation overrides such as ``[B EY1 S]`` — pass through
unchanged. Non-English text is left intact apart from Unicode cleanup.

Usage:
    from omnivoice_edu.text import preprocess_text_for_tts

    preprocess_text_for_tts("The area is $\\pi r^2$ for r = 7.")
    # -> "The area is pi r squared for r equals seven."
"""

import re
from html import unescape


# =============================================================================
# INVISIBLE UNICODE REMOVAL
# =============================================================================

_INVISIBLE_UNICODE_RE = re.compile(
    "["
    "\u200b"  # zero-width space
    "\u200e-\u200f"  # bidi marks
    "\u2028-\u2029"  # line/paragraph separators
    "\u2060-\u2064"  # word joiner, invisible math operators
    "\u00ad"  # soft hyphen
    "\u180e"  # mongolian vowel separator
    "\ufeff"  # BOM / zero-width no-break space
    "\ufff9-\ufffb"  # interlinear annotations
    "]"
)

# Preserve ZWNJ (\u200c) and ZWJ (\u200d) for Devanagari/Hindi


# =============================================================================
# MARKDOWN / HTML FLATTENING
# =============================================================================

_MD_FENCE_RE = re.compile(r"```[\s\S]*?```|~~~[\s\S]*?~~~")
_MD_IMAGE_LINK_RE = re.compile(r"!\[([^\]]*)\]\(([^)]*)\)")
_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]*)\)")
_MD_INLINE_CODE_RE = re.compile(r"`([^`]+)`")
_MD_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s*", re.MULTILINE)
_MD_UNORDERED_RE = re.compile(r"^\s*[-*+]\s+", re.MULTILINE)
_MD_ORDERED_RE = re.compile(r"^\s*\d{1,4}[.)]\s+", re.MULTILINE)
_MD_BLOCKQUOTE_RE = re.compile(r"^\s{0,3}>\s?", re.MULTILINE)
_MD_BR_TAG_RE = re.compile(r"(?i)<br\s*/?>")
_MD_HTML_TAG_RE = re.compile(r"</?[a-zA-Z][a-zA-Z0-9]*[^>]*>")
_MD_AUTOLINK_RE = re.compile(r"<https?://[^>]+>")
_MD_URL_RE = re.compile(r"https?://[^\s)>]+")
_MD_BOLD_ITALIC_RE = re.compile(r"\*\*\*(.+?)\*\*\*|___([^_]+?)___")  # ***text***
_MD_BOLD_RE = re.compile(r"\*\*(.+?)\*\*|__([^_]+?)__")
_MD_ITALIC_RE = re.compile(
    r"(?<!\w)\*([^*\n]+?)\*(?!\w)|(?<=\s)_([^_\n]+?)_(?=[\s.,;:!?]|$)"
)
_MD_STRIKETHROUGH_RE = re.compile(r"~~(.+?)~~")  # ~~text~~
_MD_TASK_LIST_RE = re.compile(r"- \[[ xX]\]\s*")  # - [x] task items
_MD_FOOTNOTE_RE = re.compile(r"\[\^\d+\]")  # [^1] footnotes
_MD_TABLE_SEP_RE = re.compile(
    r"^\|?[\s\-:]+\|[\s\-:|]+$", re.MULTILINE
)  # table separator rows
_MD_TABLE_CELL_RE = re.compile(r"\|")  # table pipe separators
_MD_DOUBLE_BACKTICK_RE = re.compile(r"``([^`]+)``")  # ``code``
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")


def _flatten_markdown(text: str) -> str:
    """Remove markdown formatting, keeping readable text."""
    text = _MD_FENCE_RE.sub(" ", text)
    text = unescape(text)
    text = _MD_BR_TAG_RE.sub(" ", text)
    text = _MD_IMAGE_LINK_RE.sub(lambda m: m.group(1).strip() or "image", text)
    text = _MD_LINK_RE.sub(lambda m: m.group(1), text)
    text = _MD_AUTOLINK_RE.sub("link", text)
    text = _EMAIL_RE.sub("email address", text)
    text = _MD_URL_RE.sub("link", text)
    text = _MD_BOLD_ITALIC_RE.sub(lambda m: m.group(1) or m.group(2), text)
    text = _MD_BOLD_RE.sub(lambda m: m.group(1) or m.group(2), text)
    text = _MD_ITALIC_RE.sub(lambda m: m.group(1) or m.group(2), text)
    text = _MD_STRIKETHROUGH_RE.sub(lambda m: m.group(1), text)
    text = _MD_DOUBLE_BACKTICK_RE.sub(lambda m: m.group(1), text)
    text = _MD_INLINE_CODE_RE.sub(lambda m: m.group(1), text)
    text = _MD_TASK_LIST_RE.sub("", text)
    text = _MD_FOOTNOTE_RE.sub("", text)
    text = _MD_TABLE_SEP_RE.sub("", text)
    text = _MD_TABLE_CELL_RE.sub(", ", text)
    text = _MD_HEADING_RE.sub("", text)
    text = _MD_UNORDERED_RE.sub("", text)
    text = _MD_ORDERED_RE.sub("", text)
    text = _MD_BLOCKQUOTE_RE.sub("", text)
    text = _MD_HTML_TAG_RE.sub(" ", text)
    return text


# =============================================================================
# UNICODE SYMBOL → SPEAKABLE TEXT
# =============================================================================

# Arrows
ARROW_MAP = {
    "\u2192": ", ",  # →
    "\u2190": ", ",  # ←
    "\u2194": ", ",  # ↔
    "\u21d2": ", ",  # ⇒
    "\u21d0": ", ",  # ⇐
    "\u27a4": ", ",  # ➤
    "\u25b6": ", ",  # ▶
    "\u25ba": ", ",  # ►
    "\u25c0": ", ",  # ◀
    "\u25c4": ", ",  # ◄
    "\u2023": ", ",  # ‣
    "\u27a1": ", ",  # ➡
    "\u2b95": ", ",  # ⮕
}

# Dashes
DASH_MAP = {
    "\u2014": ", ",  # — em dash
    "\u2013": " to ",  # – en dash (10–20 → "10 to 20")
    "\u2015": ", ",  # ― horizontal bar
    "\u2012": "-",  # ‒ figure dash
}

# Bullets — use space (not comma) so inline bullets like (•) don't produce ", "
BULLET_MAP = {
    "\u2022": " ",  # •
    "\u25aa": " ",  # ▪
    "\u25ab": " ",  # ▫
    "\u25cf": " ",  # ●
    "\u25cb": " ",  # ○
    "\u2043": " ",  # ⁃
    "\u25e6": " ",  # ◦
}

# Quotes → ASCII
QUOTE_MAP = {
    "\u201c": '"',  # "
    "\u201d": '"',  # "
    "\u2018": "'",  # '
    "\u2019": "'",  # '
    "\u201e": '"',  # „
    "\u201a": "'",  # ‚
    "\u00ab": '"',  # «
    "\u00bb": '"',  # »
}

# Math/special → speakable words
MATH_MAP = {
    "\u00d7": " times ",  # ×
    "\u00f7": " divided by ",  # ÷
    "\u00b1": " plus or minus ",  # ±
    "\u2260": " not equal to ",  # ≠
    "\u2264": " less than or equal to ",  # ≤
    "\u2265": " greater than or equal to ",  # ≥
    "\u221e": " infinity ",  # ∞
    "\u2248": " approximately ",  # ≈
    "\u2026": "...",  # …
    "\u00a9": "",  # ©
    "\u00ae": "",  # ®
    "\u2122": "",  # ™
    # Note: °, ², ³ are handled by dedicated handlers (_expand_temperatures,
    # _expand_unicode_superscripts) and are NOT in this map.
}

# ASCII math operators that TTS fumbles on
ASCII_MATH_MAP = {
    " = ": " equals ",
    " == ": " equals ",
    " != ": " not equals ",
    " !== ": " not equals ",
    " === ": " equals ",
    " >= ": " greater than or equal to ",
    " <= ": " less than or equal to ",
    " >> ": " much greater than ",
    " << ": " much less than ",
    " += ": " plus equals ",
    " -= ": " minus equals ",
    " *= ": " times equals ",
    " /= ": " divided by equals ",
    " => ": ", ",
    " -> ": ", ",
    " <> ": " not equal to ",
    " > ": " greater than ",
    " < ": " less than ",
    " + ": " plus ",
    " - ": " minus ",
}

ALL_UNICODE_MAP = {**ARROW_MAP, **DASH_MAP, **BULLET_MAP, **QUOTE_MAP, **MATH_MAP}


# =============================================================================
# ASCII MATH/OPERATOR CLEANING
# =============================================================================

# Regex for = signs in various contexts
_EQUALS_IN_ASSIGNMENT_RE = re.compile(r"(\w+)\s*=\s*(\d)")
_STANDALONE_EQUALS_RE = re.compile(r"\s+=\s+")

# Percentages: "85.5%" → "85.5 percent"
_PERCENT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%")

# Unit abbreviations → full words for TTS
_UNIT_EXPANSION = {
    "GB": "gigabytes",
    "MB": "megabytes",
    "KB": "kilobytes",
    "TB": "terabytes",
    "ms": "milliseconds",
    "ns": "nanoseconds",
    "μs": "microseconds",
    "Hz": "hertz",
    "kHz": "kilohertz",
    "MHz": "megahertz",
    "GHz": "gigahertz",
    "km": "kilometers",
    "cm": "centimeters",
    "mm": "millimeters",
    "kg": "kilograms",
    "mg": "milligrams",
    "ml": "milliliters",
    "fps": "frames per second",
    "dpi": "dots per inch",
    "px": "pixels",
    "rem": "rem",
    "em": "em",
    "pt": "points",
    "sec": "seconds",
    "min": "minutes",
    "hr": "hours",
    "Mbps": "megabits per second",
    "Gbps": "gigabits per second",
    "Kbps": "kilobits per second",
}

# Units after numbers: "2.5GB" → "2.5 gigabytes"
_UNIT_KEYS = "|".join(
    re.escape(k) for k in sorted(_UNIT_EXPANSION.keys(), key=len, reverse=True)
)
_UNIT_RE = re.compile(
    rf"(\d+(?:\.\d+)?)\s*({_UNIT_KEYS})\b",
)

# Command-line flags: "--epochs=50" → "epochs equals 50"
_CLI_FLAG_RE = re.compile(r"--(\w+)=(\S+)")


def _clean_ascii_math(text: str) -> str:
    """Replace ASCII math operators with speakable words."""
    # Multi-char operators first (order matters: >= before >)
    for pattern, replacement in ASCII_MATH_MAP.items():
        text = text.replace(pattern, replacement)

    # CLI flags: --epochs=50 → epochs equals 50
    text = _CLI_FLAG_RE.sub(r"\1 equals \2", text)

    # Variable assignment: x=5 → x equals 5
    text = _EQUALS_IN_ASSIGNMENT_RE.sub(r"\1 equals \2", text)

    # Remaining standalone = with spaces
    text = _STANDALONE_EQUALS_RE.sub(" equals ", text)

    # Percentages
    text = _PERCENT_RE.sub(r"\1 percent", text)

    # Handle bare % symbol (e.g. in quotes: the '%' sign → the percent sign)
    text = re.sub(r"['\"]%['\"]", "percent", text)  # '%' or "%" → percent
    text = text.replace("%", " percent")  # any remaining bare %

    # Units: expand abbreviations  "2.5GB" → "2.5 gigabytes"
    def _expand_unit(m):
        num = m.group(1)
        unit = m.group(2)
        expanded = _UNIT_EXPANSION.get(unit, unit)
        return f"{num} {expanded}"

    text = _UNIT_RE.sub(_expand_unit, text)

    # Ampersand
    text = re.sub(r"\s*&\s*", " and ", text)

    # Slash between words: "tokens/sec" → "tokens per second"
    def _expand_slash(m):
        word1, word2 = m.group(1), m.group(2)
        # Expand common abbreviations after slash
        abbrevs = {
            "sec": "second",
            "min": "minute",
            "hr": "hour",
            "wk": "week",
            "mo": "month",
            "yr": "year",
        }
        word2 = abbrevs.get(word2, word2)
        return f"{word1} per {word2}"

    text = re.sub(r"(\w+)/(\w+)", _expand_slash, text)

    return text


# =============================================================================
# NUMBER VERBALIZATION (English)
# =============================================================================

_ONES = {
    0: "zero",
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
}
_TEENS = {
    10: "ten",
    11: "eleven",
    12: "twelve",
    13: "thirteen",
    14: "fourteen",
    15: "fifteen",
    16: "sixteen",
    17: "seventeen",
    18: "eighteen",
    19: "nineteen",
}
_TENS = {
    20: "twenty",
    30: "thirty",
    40: "forty",
    50: "fifty",
    60: "sixty",
    70: "seventy",
    80: "eighty",
    90: "ninety",
}
_SCALES = ["", "thousand", "million", "billion", "trillion", "quadrillion"]


def _verbalize_sub_thousand(n: int, *, use_and: bool = False) -> str:
    if not 0 <= n <= 999:
        raise ValueError(f"Expected [0, 999], got {n}")
    parts: list[str] = []
    hundreds = n // 100
    rem = n % 100
    if hundreds:
        parts.append(f"{_ONES[hundreds]} hundred")
        if rem and use_and:
            parts.append("and")
    if rem:
        if rem < 10:
            parts.append(_ONES[rem])
        elif rem < 20:
            parts.append(_TEENS[rem])
        else:
            tens = (rem // 10) * 10
            ones = rem % 10
            parts.append(f"{_TENS[tens]}-{_ONES[ones]}" if ones else _TENS[tens])
    return " ".join(parts) if parts else "zero"


def _verbalize_integer_en(num_str: str, *, use_and: bool = False) -> str:
    s = num_str.replace(",", "").strip()
    if not re.fullmatch(r"\d+", s):
        raise ValueError(f"Not a plain integer: {num_str}")
    n = int(s)
    if n == 0:
        return "zero"
    groups: list[int] = []
    while n > 0:
        groups.append(n % 1000)
        n //= 1000
    if len(groups) > len(_SCALES):
        raise ValueError(f"Integer too large: {num_str}")
    parts: list[str] = []
    for scale_idx in range(len(groups) - 1, -1, -1):
        group_val = groups[scale_idx]
        if group_val == 0:
            continue
        group_words = _verbalize_sub_thousand(
            group_val,
            use_and=use_and and scale_idx == 0 and len(groups) > 1,
        )
        scale_word = _SCALES[scale_idx]
        parts.append(f"{group_words} {scale_word}".strip())
    return " ".join(parts)


def _verbalize_decimal_en(num_str: str, *, use_and: bool = False) -> str:
    s = num_str.replace(",", "").strip()
    match = re.fullmatch(r"(\d+)\.(\d+)", s)
    if not match:
        raise ValueError(f"Not a decimal: {num_str}")
    int_part, frac_part = match.groups()
    int_words = _verbalize_integer_en(int_part, use_and=use_and)
    frac_words = " ".join(_ONES[int(ch)] for ch in frac_part)
    return f"{int_words} point {frac_words}"


def _verbalize_number_en(num_str: str, *, use_and: bool = False) -> str:
    s = num_str.strip()
    if s.startswith("-"):
        inner = s[1:]
        if not inner:
            raise ValueError(f"Unsupported: {num_str}")
        return f"negative {_verbalize_number_en(inner, use_and=use_and)}"
    # Year-like 4-digit numbers (1400–2099)
    if re.fullmatch(r"\d{4}", s):
        year = int(s)
        if 1400 <= year < 2100:
            if year == 2000:
                return "two thousand"
            first_two = year // 100
            last_two = year % 100
            if 1400 <= year <= 1999:
                if last_two == 0:
                    return f"{_verbalize_integer_en(str(first_two))} hundred"
                return f"{_verbalize_integer_en(str(first_two))} {_verbalize_sub_thousand(last_two)}"
            if 2001 <= year <= 2009:
                return f"two thousand {_ONES[last_two]}"
            if 2010 <= year <= 2099:
                return f"twenty {_verbalize_sub_thousand(last_two)}"
    if re.fullmatch(r"\d[\d,]*", s):
        return _verbalize_integer_en(s, use_and=use_and)
    if re.fullmatch(r"\d[\d,]*\.\d+", s):
        return _verbalize_decimal_en(s, use_and=use_and)
    raise ValueError(f"Unsupported: {num_str}")


# =============================================================================
# CURRENCY VERBALIZATION
# =============================================================================

_CURRENCY_INFO = {
    "$": ("dollar", "dollars", "cent", "cents"),
    "£": ("pound", "pounds", "penny", "pence"),
    "€": ("euro", "euros", "cent", "cents"),
    "¥": ("yen", "yen", None, None),
    "₹": ("rupee", "rupees", "paise", "paise"),
    "¢": ("cent", "cents", None, None),
}


def _verbalize_currency_en(token: str) -> str:
    match = re.fullmatch(r"([$£€¥₹¢])(\d[\d,]*)(?:\.(\d+))?", token.strip())
    if not match:
        raise ValueError(f"Not a currency amount: {token}")
    symbol, whole_part, frac_part = match.groups()
    whole = int(whole_part.replace(",", ""))
    singular_major, plural_major, singular_minor, plural_minor = _CURRENCY_INFO[symbol]
    whole_words = _verbalize_integer_en(str(whole))
    major_unit = singular_major if whole == 1 else plural_major

    if frac_part is None or set(frac_part) == {"0"}:
        return f"{whole_words} {major_unit}"

    if len(frac_part) > 2:
        frac_words = " ".join(_ONES[int(ch)] for ch in frac_part)
        return f"{whole_words} point {frac_words} {major_unit}"

    minor = int((frac_part + "00")[:2])
    if minor == 0:
        return f"{whole_words} {major_unit}"

    if symbol in {"¥", "¢"} or singular_minor is None:
        frac_words = " ".join(_ONES[int(ch)] for ch in frac_part)
        return f"{whole_words} point {frac_words} {major_unit}"

    minor_words = _verbalize_integer_en(str(minor))
    minor_unit = singular_minor if minor == 1 else plural_minor
    if whole == 0:
        return f"{minor_words} {minor_unit}"
    return f"{whole_words} {major_unit} and {minor_words} {minor_unit}"


# =============================================================================
# TOKEN-LEVEL NUMBER/CURRENCY DETECTION
# =============================================================================

_TOKEN_RE = re.compile(
    r"""
    (?P<currency>
        (?P<symbol>[$£€¥₹¢])
        (?P<amount>\d[\d,]*(?:\.\d+)?)
    )
    |
    (?P<number>
        (?<!\w)-?\d[\d,]*(?:\.\d+)?\b
    )
    """,
    re.VERBOSE,
)


_SCALE_WORDS_RE = re.compile(
    r"([$£€¥₹¢])(\d[\d,]*(?:\.\d+)?)\s+"
    r"(hundred|thousand|million|billion|trillion|quadrillion|lakh|lakhs|crore|crores)\b",
    re.IGNORECASE,
)


def _verbalize_currency_with_scale(match: re.Match[str]) -> str:
    """Handle $3.2 billion → three point two billion dollars."""
    symbol, amount, scale = match.groups()
    _, plural_major, _, _ = _CURRENCY_INFO[symbol]
    try:
        num_words = _verbalize_number_en(amount, use_and=True)
        return f"{num_words} {scale} {plural_major}"
    except (ValueError, IndexError):
        return match.group(0)


def _verbalize_number_compact(num_str: str) -> str:
    """Verbalize a number in a TTS-friendly compact format.

    For 4+ digit numbers, the full verbalization like
    "four thousand three hundred and twenty-eight" can confuse TTS models
    into producing garbled output (e.g. splitting into separate numbers).

    This function produces a simpler format that TTS models handle reliably:
    - 4328 → "four thousand three hundred twenty eight" (no "and", no hyphens)
    - 428 → "four hundred twenty eight" (compact, no "and")
    - 28 → "twenty eight" (compact, no hyphen)
    - 5.5 → "five point five"
    """
    s = num_str.strip()
    is_negative = s.startswith("-")
    if is_negative:
        s = s[1:]
    if not s:
        return num_str

    # Handle decimals
    if "." in s:
        int_part, frac_part = s.split(".", 1)
        int_clean = int_part.replace(",", "")
        if int_clean:
            int_words = _verbalize_number_compact(int_clean)
        else:
            int_words = ""
        frac_words = " ".join(_ONES[int(ch)] for ch in frac_part)
        result = f"{int_words} point {frac_words}".strip()
        return f"negative {result}" if is_negative else result

    # Clean integer
    s = s.replace(",", "")
    if not re.fullmatch(r"\d+", s):
        return num_str
    n = int(s)

    if is_negative:
        return f"negative {_verbalize_number_compact(s)}"

    # 4+ digit numbers: use group verbalization for TTS reliability
    if n >= 1000:
        # Split into groups of 3 digits (thousands, millions, etc.)
        groups: list[int] = []
        temp = n
        while temp > 0:
            groups.append(temp % 1000)
            temp //= 1000
        if len(groups) > len(_SCALES):
            # Too large, fall back to digit-by-digit
            return " ".join(_ONES[int(ch)] for ch in s)
        parts: list[str] = []
        for scale_idx in range(len(groups) - 1, -1, -1):
            group_val = groups[scale_idx]
            if group_val == 0:
                continue
            group_words = _verbalize_sub_thousand_compact(group_val)
            scale_word = _SCALES[scale_idx]
            parts.append(f"{group_words} {scale_word}".strip())
        return " ".join(parts)

    # 1-999: use compact sub-thousand verbalization
    return _verbalize_sub_thousand_compact(n)


def _verbalize_sub_thousand_compact(n: int) -> str:
    """Verbalize 0-999 in compact TTS-friendly format (no "and", no hyphens)."""
    if n == 0:
        return "zero"
    parts: list[str] = []
    hundreds = n // 100
    rem = n % 100
    if hundreds:
        parts.append(f"{_ONES[hundreds]} hundred")
    if rem:
        if rem < 10:
            parts.append(_ONES[rem])
        elif rem < 20:
            parts.append(_TEENS[rem])
        else:
            tens = (rem // 10) * 10
            ones = rem % 10
            if ones:
                parts.append(f"{_TENS[tens]} {_ONES[ones]}")
            else:
                parts.append(_TENS[tens])
    return " ".join(parts)


def _auto_verbalize_numbers(text: str) -> str:
    """Find and verbalize standalone numbers and currency amounts."""
    # First pass: handle currency + scale word (e.g. "$3.2 billion")
    text = _SCALE_WORDS_RE.sub(_verbalize_currency_with_scale, text)

    # Second pass: handle remaining currencies and standalone numbers
    def repl(match: re.Match[str]) -> str:
        if match.group("currency") is not None:
            token = match.group("currency")
            try:
                return _verbalize_currency_en(token)
            except (ValueError, IndexError):
                return token
        token = match.group("number")
        try:
            s = token.strip().lstrip("-")
            raw_int = s.split(".", 1)[0].replace(",", "")
            if not re.fullmatch(r"\d+", raw_int):
                return token
            if int(raw_int) >= 10**15:
                return token
            # Use compact verbalization for TTS reliability
            return _verbalize_number_compact(token)
        except (ValueError, IndexError):
            return token

    return _TOKEN_RE.sub(repl, text)


# =============================================================================
# LaTeX / MATH NOTATION HANDLING
# =============================================================================

# Greek letters
_LATEX_GREEK = {
    r"\alpha": "alpha",
    r"\beta": "beta",
    r"\gamma": "gamma",
    r"\delta": "delta",
    r"\epsilon": "epsilon",
    r"\zeta": "zeta",
    r"\eta": "eta",
    r"\theta": "theta",
    r"\iota": "iota",
    r"\kappa": "kappa",
    r"\lambda": "lambda",
    r"\mu": "mu",
    r"\nu": "nu",
    r"\xi": "xi",
    r"\pi": "pi",
    r"\rho": "rho",
    r"\sigma": "sigma",
    r"\tau": "tau",
    r"\upsilon": "upsilon",
    r"\phi": "phi",
    r"\varphi": "phi",
    r"\chi": "chi",
    r"\psi": "psi",
    r"\omega": "omega",
    r"\Alpha": "Alpha",
    r"\Beta": "Beta",
    r"\Gamma": "Gamma",
    r"\Delta": "Delta",
    r"\Theta": "Theta",
    r"\Lambda": "Lambda",
    r"\Pi": "Pi",
    r"\Sigma": "Sigma",
    r"\Phi": "Phi",
    r"\Psi": "Psi",
    r"\Omega": "Omega",
}

# LaTeX operators and symbols
_LATEX_SYMBOLS = {
    r"\rightarrow": " implies ",
    r"\Rightarrow": " implies ",
    r"\leftarrow": " from ",
    r"\Leftarrow": " from ",
    r"\leftrightarrow": " if and only if ",
    r"\Leftrightarrow": " if and only if ",
    r"\leq": " less than or equal to ",
    r"\geq": " greater than or equal to ",
    r"\neq": " not equal to ",
    r"\approx": " approximately ",
    r"\equiv": " is equivalent to ",
    r"\sim": " is similar to ",
    r"\propto": " is proportional to ",
    r"\infty": " infinity ",
    r"\partial": " partial ",
    r"\nabla": " nabla ",
    r"\therefore": " therefore ",
    r"\because": " because ",
    r"\forall": " for all ",
    r"\exists": " there exists ",
    r"\in": " in ",
    r"\notin": " not in ",
    r"\subset": " is a subset of ",
    r"\supset": " is a superset of ",
    r"\cup": " union ",
    r"\cap": " intersection ",
    r"\times": " times ",
    r"\div": " divided by ",
    r"\cdot": " times ",
    r"\pm": " plus or minus ",
    r"\mp": " minus or plus ",
    r"\ldots": "...",
    r"\cdots": "...",
    r"\quad": " ",
    r"\qquad": " ",
    r"\text": "",
    r"\mathrm": "",
    r"\mathbf": "",
    r"\mathit": "",
    r"\left": "",
    r"\right": "",
    r"\big": "",
    r"\Big": "",
    r"\bigg": "",
    r"\Bigg": "",
}

# Display/inline math delimiters
_LATEX_DISPLAY_RE = re.compile(r"\$\$(.*?)\$\$", re.DOTALL)
_LATEX_INLINE_RE = re.compile(r"\$([^$]+?)\$")
# \( ... \) and \[ ... \]
_LATEX_PAREN_INLINE_RE = re.compile(r"\\\((.*?)\\\)")
_LATEX_BRACKET_DISPLAY_RE = re.compile(r"\\\[(.*?)\\\]", re.DOTALL)

# LaTeX commands with arguments
_NESTED_BRACE = r"(?:[^{}]|\{[^}]*\})*"
_LATEX_FRAC_RE = re.compile(
    r"\\frac\s*\{(" + _NESTED_BRACE + r")\}\s*\{(" + _NESTED_BRACE + r")\}"
)
_LATEX_SQRT_RE = re.compile(
    r"\\sqrt\s*(?:\[([^\]]*)\])?\s*\{(" + _NESTED_BRACE + r")\}"
)
_LATEX_SUM_RE = re.compile(r"\\sum\s*(?:_\{([^}]*)\})?\s*(?:\^\{([^}]*)\})?")
_LATEX_INT_RE = re.compile(r"\\int\s*(?:_\{([^}]*)\})?\s*(?:\^\{([^}]*)\})?")
_LATEX_LIM_RE = re.compile(r"\\lim\s*(?:_\{([^}]*)\})?")
_LATEX_SUPERSCRIPT_RE = re.compile(r"\^\{([^}]*)\}")
_LATEX_SUBSCRIPT_RE = re.compile(r"_\{([^}]*)\}")
_LATEX_CARET_SUPER_RE = re.compile(r"\^(\w)")
_LATEX_UNDERSCORE_SUB_RE = re.compile(r"_(\w)")
_LATEX_BRACES_RE = re.compile(r"\{([^}]*)\}")
_LATEX_OVERLINE_RE = re.compile(r"\\overline\s*\{([^}]*)\}")
_LATEX_HAT_RE = re.compile(r"\\hat\s*\{([^}]*)\}")
_LATEX_VEC_RE = re.compile(r"\\vec\s*\{([^}]*)\}")
_LATEX_BAR_RE = re.compile(r"\\bar\s*\{([^}]*)\}")


def _superscript_to_words(s: str) -> str:
    """Convert a superscript expression to spoken words."""
    s = s.strip()
    if s == "2":
        return " squared"
    elif s == "3":
        return " cubed"
    elif s == "n":
        return " to the n"
    elif s == "-1":
        return " inverse"
    return f" to the power of {s}"


def _clean_latex_expression(expr: str) -> str:
    """Convert a LaTeX math expression to spoken English."""
    text = expr.strip()

    # Fractions: \frac{a}{b} → "a over b"
    text = _LATEX_FRAC_RE.sub(lambda m: f"{m.group(1)} over {m.group(2)}", text)

    # Square roots: \sqrt{x} → "square root of x", \sqrt[3]{x} → "cube root of x"
    def _sqrt_repl(m):
        idx = m.group(1)
        val = m.group(2)
        if idx is None:
            return f"square root of {val}"
        elif idx == "3":
            return f"cube root of {val}"
        else:
            return f"{idx}th root of {val}"

    text = _LATEX_SQRT_RE.sub(_sqrt_repl, text)

    # Sum: \sum_{i=0}^{n} → "sum from i equals zero to n"
    def _sum_repl(m):
        lower, upper = m.group(1), m.group(2)
        result = "sum"
        if lower:
            result += f" from {lower}"
        if upper:
            result += f" to {upper}"
        return result

    text = _LATEX_SUM_RE.sub(_sum_repl, text)

    # Integral: \int_{a}^{b} → "integral from a to b"
    def _int_repl(m):
        lower, upper = m.group(1), m.group(2)
        result = "integral"
        if lower:
            result += f" from {lower}"
        if upper:
            result += f" to {upper}"
        return result

    text = _LATEX_INT_RE.sub(_int_repl, text)

    # Limit: \lim_{x \to 0} → "limit as x approaches zero"
    def _lim_repl(m):
        cond = m.group(1)
        if cond:
            cond = cond.replace(r"\to", " approaches ")
            return f"limit as {cond}"
        return "limit"

    text = _LATEX_LIM_RE.sub(_lim_repl, text)

    # Overline, hat, vec, bar
    text = _LATEX_OVERLINE_RE.sub(lambda m: f"{m.group(1)} bar", text)
    text = _LATEX_HAT_RE.sub(lambda m: f"{m.group(1)} hat", text)
    text = _LATEX_VEC_RE.sub(lambda m: f"vector {m.group(1)}", text)
    text = _LATEX_BAR_RE.sub(lambda m: f"{m.group(1)} bar", text)

    # Greek letters and symbols (sorted by length, longest first)
    for cmd in sorted(_LATEX_GREEK.keys(), key=len, reverse=True):
        text = text.replace(cmd, f" {_LATEX_GREEK[cmd]} ")
    for cmd in sorted(_LATEX_SYMBOLS.keys(), key=len, reverse=True):
        text = text.replace(cmd, _LATEX_SYMBOLS[cmd])

    # Superscripts: ^{2} → " squared", ^{n} → " to the n"
    text = _LATEX_SUPERSCRIPT_RE.sub(lambda m: _superscript_to_words(m.group(1)), text)
    # Single char superscripts: ^2 → " squared"
    text = _LATEX_CARET_SUPER_RE.sub(lambda m: _superscript_to_words(m.group(1)), text)

    # Subscripts: _{i} → " sub i"
    text = _LATEX_SUBSCRIPT_RE.sub(lambda m: f" sub {m.group(1)}", text)
    text = _LATEX_UNDERSCORE_SUB_RE.sub(lambda m: f" sub {m.group(1)}", text)

    # Remove remaining braces
    text = _LATEX_BRACES_RE.sub(lambda m: m.group(1), text)

    # Clean up remaining backslash commands we missed
    text = re.sub(r"\\[a-zA-Z]+", "", text)

    # Handle plain minus in math context: " - " → " minus "
    text = text.replace(" - ", " minus ")

    # Clean up extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def _process_latex(text: str) -> str:
    """Find and convert all LaTeX math notation in text to spoken English."""
    # Process display math first ($$...$$), then inline ($...$)
    text = _LATEX_BRACKET_DISPLAY_RE.sub(
        lambda m: _clean_latex_expression(m.group(1)), text
    )
    text = _LATEX_DISPLAY_RE.sub(lambda m: _clean_latex_expression(m.group(1)), text)
    text = _LATEX_PAREN_INLINE_RE.sub(
        lambda m: _clean_latex_expression(m.group(1)), text
    )
    text = _LATEX_INLINE_RE.sub(lambda m: _clean_latex_expression(m.group(1)), text)

    # Also handle bare LaTeX commands outside of $ delimiters
    for cmd in sorted(_LATEX_GREEK.keys(), key=len, reverse=True):
        text = text.replace(cmd, f" {_LATEX_GREEK[cmd]} ")
    for cmd in sorted(_LATEX_SYMBOLS.keys(), key=len, reverse=True):
        text = text.replace(cmd, _LATEX_SYMBOLS[cmd])

    # Handle bare \frac, \sqrt etc. outside math delimiters
    text = _LATEX_FRAC_RE.sub(lambda m: f"{m.group(1)} over {m.group(2)}", text)
    text = _LATEX_SQRT_RE.sub(
        lambda m: (
            f"square root of {m.group(2)}"
            if m.group(1) is None
            else f"{m.group(1)}th root of {m.group(2)}"
        ),
        text,
    )

    return text


# =============================================================================
# UNICODE SUPERSCRIPT / SUBSCRIPT HANDLING
# =============================================================================

_UNICODE_SUPERSCRIPTS = {
    "\u2070": "0",
    "\u00b9": "1",
    "\u00b2": "2",
    "\u00b3": "3",
    "\u2074": "4",
    "\u2075": "5",
    "\u2076": "6",
    "\u2077": "7",
    "\u2078": "8",
    "\u2079": "9",
    "\u207f": "n",
    "\u207a": "+",
    "\u207b": "-",
    "\u207c": "=",
}

_UNICODE_SUBSCRIPTS = {
    "\u2080": "0",
    "\u2081": "1",
    "\u2082": "2",
    "\u2083": "3",
    "\u2084": "4",
    "\u2085": "5",
    "\u2086": "6",
    "\u2087": "7",
    "\u2088": "8",
    "\u2089": "9",
    "\u208a": "+",
    "\u208b": "-",
    "\u208c": "=",
}

_SUPERSCRIPT_CHARS = "".join(_UNICODE_SUPERSCRIPTS.keys())
_SUBSCRIPT_CHARS = "".join(_UNICODE_SUBSCRIPTS.keys())

_UNICODE_SUPER_RE = re.compile(f"([A-Za-z0-9]?)([{re.escape(_SUPERSCRIPT_CHARS)}]+)")
_UNICODE_SUB_RE = re.compile(f"([A-Za-z0-9]?)([{re.escape(_SUBSCRIPT_CHARS)}]+)")

# Caret notation for superscripts in plain text: x^2, 10^5
_CARET_SUPER_RE = re.compile(r"(\w)\^(\d+)\b")


def _expand_unicode_superscripts(text: str) -> str:
    """Convert unicode superscript characters to spoken words."""

    def _repl(m):
        base = m.group(1)
        chars = m.group(2)
        digits = "".join(_UNICODE_SUPERSCRIPTS.get(c, c) for c in chars)
        return f"{base}{_superscript_to_words(digits)}"

    text = _UNICODE_SUPER_RE.sub(_repl, text)
    return text


def _expand_unicode_subscripts(text: str) -> str:
    """Convert unicode subscript characters: H₂O → H 2 O."""

    def _repl(m):
        base = m.group(1)
        chars = m.group(2)
        digits = "".join(_UNICODE_SUBSCRIPTS.get(c, c) for c in chars)
        return f"{base} {digits} "

    text = _UNICODE_SUB_RE.sub(_repl, text)
    return text


def _expand_caret_notation(text: str) -> str:
    """Convert caret notation: x^2 → x squared, 10^5 → 10 to the power of 5."""

    def _repl(m):
        base = m.group(1)
        exp = m.group(2)
        return f"{base}{_superscript_to_words(exp)}"

    text = _CARET_SUPER_RE.sub(_repl, text)
    return text


# =============================================================================
# UNICODE FRACTIONS
# =============================================================================

_UNICODE_FRACTIONS = {
    "\u00bd": "one half",  # ½
    "\u2153": "one third",  # ⅓
    "\u2154": "two thirds",  # ⅔
    "\u00bc": "one quarter",  # ¼
    "\u00be": "three quarters",  # ¾
    "\u2155": "one fifth",  # ⅕
    "\u2156": "two fifths",  # ⅖
    "\u2157": "three fifths",  # ⅗
    "\u2158": "four fifths",  # ⅘
    "\u2159": "one sixth",  # ⅙
    "\u215a": "five sixths",  # ⅚
    "\u2150": "one seventh",  # ⅐
    "\u215b": "one eighth",  # ⅛
    "\u215c": "three eighths",  # ⅜
    "\u215d": "five eighths",  # ⅝
    "\u215e": "seven eighths",  # ⅞
    "\u2151": "one ninth",  # ⅑
    "\u2152": "one tenth",  # ⅒
}


def _expand_fractions(text: str) -> str:
    """Replace unicode fraction characters with spoken words."""
    for char, word in _UNICODE_FRACTIONS.items():
        text = text.replace(char, f" {word} ")
    return text


# =============================================================================
# DATE / TIME / PHONE / ORDINAL HANDLING
# =============================================================================

_ORDINAL_MAP = {
    "1": "first",
    "2": "second",
    "3": "third",
    "4": "fourth",
    "5": "fifth",
    "6": "sixth",
    "7": "seventh",
    "8": "eighth",
    "9": "ninth",
    "10": "tenth",
    "11": "eleventh",
    "12": "twelfth",
    "13": "thirteenth",
    "14": "fourteenth",
    "15": "fifteenth",
    "16": "sixteenth",
    "17": "seventeenth",
    "18": "eighteenth",
    "19": "nineteenth",
    "20": "twentieth",
    "21": "twenty-first",
    "22": "twenty-second",
    "23": "twenty-third",
    "24": "twenty-fourth",
    "25": "twenty-fifth",
    "26": "twenty-sixth",
    "27": "twenty-seventh",
    "28": "twenty-eighth",
    "29": "twenty-ninth",
    "30": "thirtieth",
    "31": "thirty-first",
}

_MONTH_NAMES = {
    1: "January",
    2: "February",
    3: "March",
    4: "April",
    5: "May",
    6: "June",
    7: "July",
    8: "August",
    9: "September",
    10: "October",
    11: "November",
    12: "December",
}

# Ordinals: 1st, 2nd, 3rd, 4th etc.
_ORDINAL_RE = re.compile(r"\b(\d{1,2})(st|nd|rd|th)\b", re.IGNORECASE)

# Dates: MM/DD/YYYY or DD-MM-YYYY or DD/MM/YYYY
_DATE_SLASH_RE = re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b")
_DATE_DASH_RE = re.compile(r"\b(\d{1,2})-(\d{1,2})-(\d{4})\b")

# Time: 3:45 PM, 14:30, 2:00pm
_TIME_RE = re.compile(r"\b(\d{1,2}):(\d{2})\s*(AM|PM|am|pm|a\.m\.|p\.m\.)?\b")

# Phone numbers: +1-555-123-4567 or (555) 123-4567
_PHONE_RE = re.compile(r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")

# Temperature: -40°C, 98.6°F
_TEMP_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s*°\s*([CFcf])\b")


def _expand_ordinals(text: str) -> str:
    """Convert ordinal numbers: 1st → first, 2nd → second."""

    def _repl(m):
        num = m.group(1)
        return _ORDINAL_MAP.get(num, f"{num}th")

    return _ORDINAL_RE.sub(_repl, text)


def _expand_dates(text: str) -> str:
    """Convert date formats to spoken English."""

    def _date_repl(m):
        p1, p2, year_str = int(m.group(1)), int(m.group(2)), m.group(3)
        # Determine if MM/DD/YYYY or DD/MM/YYYY
        # If first number > 12, it's DD/MM/YYYY
        if p1 > 12 and 1 <= p2 <= 12:
            day, month = p1, p2
        elif p2 > 12 and 1 <= p1 <= 12:
            month, day = p1, p2
        elif 1 <= p1 <= 12 and 1 <= p2 <= 31:
            month, day = p1, p2  # Default: MM/DD/YYYY
        else:
            return m.group(0)

        if month not in _MONTH_NAMES or day < 1 or day > 31:
            return m.group(0)

        day_word = _ORDINAL_MAP.get(str(day), f"{day}th")
        try:
            year_words = _verbalize_number_en(year_str)
        except ValueError:
            year_words = year_str
        return f"{_MONTH_NAMES[month]} {day_word}, {year_words}"

    text = _DATE_SLASH_RE.sub(_date_repl, text)
    text = _DATE_DASH_RE.sub(_date_repl, text)
    return text


def _expand_times(text: str) -> str:
    """Convert time formats: 3:45 PM → three forty-five PM."""

    def _repl(m):
        hour, minute, period = int(m.group(1)), int(m.group(2)), m.group(3)
        try:
            hour_w = _verbalize_integer_en(str(hour))
        except ValueError:
            return m.group(0)

        if minute == 0:
            min_w = "o'clock" if not period else ""
        elif minute < 10:
            min_w = f"oh {_verbalize_integer_en(str(minute))}"
        else:
            try:
                min_w = _verbalize_integer_en(str(minute))
            except ValueError:
                return m.group(0)

        parts = [hour_w, min_w]
        if period:
            parts.append(period.replace(".", "").upper())
        return " ".join(p for p in parts if p)

    return _TIME_RE.sub(_repl, text)


def _expand_phone_numbers(text: str) -> str:
    """Convert phone numbers to digit-by-digit reading."""

    def _repl(m):
        digits = re.sub(r"[^\d]", "", m.group(0))
        return " ".join(_ONES.get(int(d), d) for d in digits)

    return _PHONE_RE.sub(_repl, text)


def _expand_temperatures(text: str) -> str:
    """Convert temperature: -40°C → minus forty degrees Celsius."""

    def _repl(m):
        num_str, unit = m.group(1), m.group(2).upper()
        try:
            num_words = _verbalize_number_en(num_str)
        except ValueError:
            num_words = num_str
        unit_word = "Celsius" if unit == "C" else "Fahrenheit"
        return f"{num_words} degrees {unit_word}"

    text = _TEMP_RE.sub(_repl, text)

    # Handle bare degree symbol (not followed by C/F)
    text = re.sub(r"(\d)\s*°(?![CFcf])", r"\1 degrees", text)

    return text


# =============================================================================
# EMOJI HANDLING
# =============================================================================

# Common emoji to speakable text mappings
_EMOJI_MAP = {
    "😀": "",
    "😃": "",
    "😄": "",
    "😁": "",
    "😆": "",
    "😅": "",
    "🤣": "",
    "😂": "",
    "🙂": "",
    "🙃": "",
    "😉": "",
    "😊": "",
    "😇": "",
    "🥰": "",
    "😍": "",
    "😘": "",
    "😗": "",
    "😚": "",
    "😙": "",
    "😋": "",
    "😛": "",
    "😜": "",
    "🤪": "",
    "😝": "",
    "🤑": "",
    "🤗": "",
    "🤭": "",
    "🤫": "",
    "🤔": "",
    "🤐": "",
    "🤨": "",
    "😐": "",
    "😑": "",
    "😶": "",
    "😏": "",
    "😒": "",
    "🙄": "",
    "😬": "",
    "🤥": "",
    "😌": "",
    "😔": "",
    "😪": "",
    "🤤": "",
    "😴": "",
    "😷": "",
    "🤒": "",
    "🤕": "",
    "🤢": "",
    "🤮": "",
    "🥵": "",
    "🥶": "",
    "🥴": "",
    "😵": "",
    "🤯": "",
    "🤠": "",
    "🥳": "",
    "😎": "",
    "🤓": "",
    "🧐": "",
    "😕": "",
    "😟": "",
    "🙁": "",
    "😮": "",
    "😯": "",
    "😲": "",
    "😳": "",
    "🥺": "",
    "😦": "",
    "😧": "",
    "😨": "",
    "😰": "",
    "😥": "",
    "😢": "",
    "😭": "",
    "😱": "",
    "😖": "",
    "😣": "",
    "😞": "",
    "😓": "",
    "😩": "",
    "😫": "",
    "🥱": "",
    "😤": "",
    "😡": "",
    "😠": "",
    "🤬": "",
    "👍": "",
    "👎": "",
    "👏": "",
    "🙏": "",
    "🎉": "",
    "🎊": "",
    "❤️": "",
    "💔": "",
    "💯": "",
    "✅": "",
    "❌": "",
    "⭐": "",
    "🌟": "",
    "💪": "",
    "🔥": "",
    "📈": "",
    "📉": "",
    "📊": "",
    "🎵": "",
    "🎶": "",
    "💡": "",
    "⚡": "",
    "🚀": "",
    "🏆": "",
    "🤝": "",
    "👀": "",
    "🗣️": "",
    "💬": "",
    "📝": "",
}

# Regex to match common emoji ranges
_EMOJI_RE = re.compile(
    "["
    "\U0001f600-\U0001f64f"  # emoticons
    "\U0001f300-\U0001f5ff"  # symbols & pictographs
    "\U0001f680-\U0001f6ff"  # transport & map
    "\U0001f1e0-\U0001f1ff"  # flags
    "\U00002702-\U000027b0"  # dingbats
    "\U0001f900-\U0001f9ff"  # supplemental symbols
    "\U0001fa00-\U0001fa6f"  # chess symbols
    "\U0001fa70-\U0001faff"  # symbols extended-A
    "\U00002600-\U000026ff"  # misc symbols
    "]+",
    flags=re.UNICODE,
)


def _remove_emojis(text: str) -> str:
    """Remove emojis that TTS cannot handle."""
    # First replace known emojis with their mapped text (currently all empty)
    for emoji, replacement in _EMOJI_MAP.items():
        text = text.replace(emoji, replacement)
    # Remove any remaining emojis
    text = _EMOJI_RE.sub(" ", text)
    return text


# =============================================================================
# PUNCTUATION NORMALIZATION
# =============================================================================

_LINE_BREAK_RE = re.compile(r"(?:\r\n|\r|\n)+")
_REPEATED_PUNCT_RE = re.compile(r"([!?])\1+")
_ASCII_ELLIPSIS_RE = re.compile(r"\.{3,}")
_UNICODE_HYPHEN_RE = re.compile("[\u2010\u2011]")
_MULTI_HYPHEN_RE = re.compile(r"[\u002d\u2013\u2212]{2,}")
# Sentence-final marks, including Devanagari danda (\u0964, \u0965), Arabic
# question mark and CJK full stops, so no stray "." is appended to them.
_TERMINAL_PUNCT = ".!?\u2026\u0964\u0965\u061f\u3002\uff01\uff1f"

# OmniVoice inline control tags, protected from every cleaning step:
#   - non-verbal sounds: [laughter], [sigh], [question-en], [surprise-ah], ...
#   - CMU pronunciation overrides: [B EY1 S] (uppercase phonemes with at
#     least one stress digit, so ordinary bracketed text is not matched)
_PARA_TAG_RE = re.compile(
    r"(?i:\[(?:laughter|sigh|confirmation-en|question-(?:en|ah|oh|ei|yi)"
    r"|surprise-(?:ah|oh|wa|yo)|dissatisfaction-hnn)\])"
    r"|\[(?=[^\]]*\d)[A-Z]{1,2}[0-2]?(?: [A-Z]{1,2}[0-2]?)+\]"
)


# =============================================================================
# LIST / BULLET FORMATTING
# =============================================================================

_BULLET_LINE_RE = re.compile(
    r"^[\u2192\u2190\u2194\u27a4\u25b6\u25ba\u25c0\u25c4\u2023\u27a1\u2b95"
    r"\u2022\u25aa\u25ab\u25cf\u25cb\u2043\u25e6"
    r"\*\u2013\u2014\u2015\>]+"
    r"|-\s+"  # hyphen bullet only when followed by whitespace (not -15)
    r"|>\s*",
)

_NUMBERED_LIST_RE = re.compile(r"^\d{1,2}[.)]\s+|^\([a-zA-Z0-9]{1,3}\)\s+")

# Instruction prefixes to strip
_INSTRUCTION_PREFIX_RE = re.compile(
    r"^(Question\s*[:\.]\s*|Answer\s*[:\.]\s*|"
    r"Q\s*\d*\s*[:\.]\s*|A\s*\d*\s*[:\.]\s*|"
    r"Exercise\s*\d*\s*[:\.]\s*|"
    r"Activity\s*\d*\s*[:\.]\s*|"
    r"Task\s*\d*\s*[:\.]\s*|"
    r"Problem\s*\d*\s*[:\.]\s*|"
    r"Example\s*\d*\s*[:\.]\s*|"
    r"Note\s*[:\.]\s*|"
    r"Hint\s*[:\.]\s*)",
    re.IGNORECASE,
)


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================


def preprocess_text_for_tts(text: str) -> str:
    """Preprocess text before TTS synthesis.

    This is the single entry point: the TTS server runs every request
    through it, so all generation paths receive the same clean text.

    Pipeline:
      1. Protect OmniVoice control tags ([laughter], [B EY1 S], etc.)
      2. Process LaTeX/math notation → spoken English
      3. Flatten Markdown/HTML formatting
      4. Remove invisible Unicode (preserve ZWNJ/ZWJ for Devanagari)
      5. Replace Unicode symbols with speakable text
      6. Expand Unicode superscripts/subscripts/fractions and caret notation
      7. Remove emojis
      8. Expand dates, times, temperatures, ordinals, phone numbers
      9. Clean ASCII math operators (=, +, %, etc.)
      10. Strip list bullets and prefixes, then collapse line breaks
      11. Clean identifier and code artifacts
      12. Verbalize numbers and currencies to English words
      13. Normalize punctuation
      14. Final cleanup and terminal punctuation
      15. Restore protected tags

    Args:
        text: Raw input text (may contain Markdown, math, numbers, etc.)

    Returns:
        Cleaned text ready for TTS synthesis.
    """
    if not text or not text.strip():
        return text

    # Step 1: Protect OmniVoice control tags
    protected = {}
    counter = 0

    def _protect(match):
        nonlocal counter
        tag = match.group(0)
        key = f"PTAGHOLDER{counter}"
        protected[key] = tag
        counter += 1
        return key

    text = _PARA_TAG_RE.sub(_protect, text)

    # Step 2: Process LaTeX/math notation BEFORE markdown flattening
    # (so $ delimiters are handled before markdown processing)
    text = _process_latex(text)

    # Step 3: Flatten markdown (code blocks, links, headings, bold, italic, URLs)
    text = _flatten_markdown(text)

    # Step 4: Remove invisible Unicode (preserve ZWNJ/ZWJ for Hindi)
    text = _INVISIBLE_UNICODE_RE.sub("", text)

    # Step 5: Replace Unicode symbols → speakable text
    for char, replacement in ALL_UNICODE_MAP.items():
        text = text.replace(char, replacement)

    # Step 6: Expand unicode superscripts, subscripts, fractions, and caret notation
    text = _expand_unicode_superscripts(text)
    text = _expand_unicode_subscripts(text)
    text = _expand_fractions(text)
    text = _expand_caret_notation(text)

    # Step 7: Remove emojis
    text = _remove_emojis(text)

    # Step 8: Expand dates, times, temperatures BEFORE ascii math cleaning
    # (so slashes in dates like 01/15/2024 aren't treated as division)
    text = _expand_dates(text)
    text = _expand_times(text)
    text = _expand_temperatures(text)
    text = _expand_ordinals(text)
    text = _expand_phone_numbers(text)

    # Step 9: Clean ASCII math operators (=, >=, <=, %, &, etc.)
    text = _clean_ascii_math(text)

    # Step 10: Handle lists/bullets BEFORE collapsing line breaks, while
    # the line structure still exists. Strip instruction prefixes first.
    text = _INSTRUCTION_PREFIX_RE.sub("", text)

    # Clean bullet lines - process while newlines still exist
    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        stripped = _BULLET_LINE_RE.sub("", stripped)
        stripped = _NUMBERED_LIST_RE.sub(" ", stripped).strip()
        if stripped:
            cleaned_lines.append(stripped)
    text = " ".join(cleaned_lines)

    # Collapse any remaining line breaks
    text = _LINE_BREAK_RE.sub(" ", text)

    # Step 11: Replace underscores in identifiers with spaces for TTS
    text = re.sub(r"(\w)_(\w)", r"\1 \2", text)

    # Clean programming artifacts: model.fit() → model fit, train.py → train py
    text = re.sub(r"(\w)\.(\w+)\(\)", r"\1 \2", text)  # method calls
    text = re.sub(
        r"(\w+)\.(py|js|ts|cpp|java|rb|go)\b", r"\1 \2", text
    )  # file extensions

    # Step 12: Verbalize numbers and currencies. A bare ₹ not followed by a
    # digit (e.g. ₹x after LaTeX processing) is spoken as "rupees".
    text = re.sub(r"₹(?!\d)", "rupees ", text)
    text = _auto_verbalize_numbers(text)

    # Step 13: Normalize punctuation
    text = _UNICODE_HYPHEN_RE.sub("-", text)
    text = _ASCII_ELLIPSIS_RE.sub("...", text)
    text = _REPEATED_PUNCT_RE.sub(r"\1", text)
    text = _MULTI_HYPHEN_RE.sub(", ", text)

    # Fix punctuation spacing
    text = re.sub(r"([.!?])\1+", r"\1", text)
    text = re.sub(r"([,;:])\s*\1+", r"\1", text)
    text = re.sub(r"\s+([.!?,;:])", r"\1", text)
    text = re.sub(r"([.!?,;:])([A-Za-z\u0900-\u097F])", r"\1 \2", text)

    # Step 14: Final cleanup
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    text = re.sub(r"[\u200b\ufeff\u200e\u200f]", "", text)
    # Clean up empty/whitespace-only parentheses: "( )" or "()" → ""
    text = re.sub(r"\(\s*\)", "", text)
    # Clean up quoted percent: '%' or "%" → percent
    text = re.sub(r"['\"]percent['\"]", "percent", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"^[,;:\s]+", "", text)
    text = re.sub(r"[,;:\s]+$", "", text)

    # Ensure terminal punctuation
    if text and text[-1] not in _TERMINAL_PUNCT:
        text += "."

    # Step 15: Restore protected tags
    for key, value in protected.items():
        text = text.replace(key, value)

    return text
