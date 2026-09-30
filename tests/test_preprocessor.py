"""Tests for the text preprocessor.

The first table mirrors the "Text preprocessing" examples in README.md, so
the documentation can't drift from the real behaviour.

Run: python -m tests.test_preprocessor
"""

from omnivoice_edu.text import preprocess_text_for_tts as prep

# (input, expected output) — keep in sync with README.md
README_EXAMPLES = [
    ("3x + 5 = 20", "3x plus five equals twenty."),
    ("x² + y² = z²", "x squared plus y squared equals z squared."),
    (r"$\pi r^2$", "pi r squared."),
    (r"$\frac{a}{b}$", "a over b."),
    ("₹3.5 crore", "three point five crore rupees."),
    ("₹2,500", "two thousand five hundred rupees."),
    ("$1,250", "one thousand two hundred fifty dollars."),
    ("12.5%", "twelve point five percent."),
    ("35°C", "thirty-five degrees Celsius."),
    ("H₂O", "H two O."),
    ("01/15/2024", "January fifteenth, twenty twenty-four."),
    ("3:45 PM", "three forty-five PM."),
    ("1st, 2nd, 3rd", "first, second, third."),
    ("**bold** and [a link](https://example.com)", "bold and a link."),
    ("Great job 🎉", "Great job."),
]


def test_readme_examples():
    for text, expected in README_EXAMPLES:
        got = prep(text)
        assert got == expected, f"{text!r}: expected {expected!r}, got {got!r}"


def test_docstring_example():
    got = prep(r"The area is $\pi r^2$ for r = 7.")
    assert got == "The area is pi r squared for r equals seven.", got


def test_large_numbers_are_grouped():
    assert prep("4328 students") == "four thousand three hundred twenty eight students."


def test_nonverbal_tags_preserved():
    for tag in ("[laughter]", "[sigh]", "[question-en]", "[surprise-ah]"):
        got = prep(f"{tag} You got me.")
        assert got.startswith(tag), got


def test_cmu_pronunciation_preserved():
    got = prep("He plays the [B EY1 S] guitar.")
    assert got == "He plays the [B EY1 S] guitar.", got


def test_plain_brackets_not_treated_as_phonemes():
    # Only uppercase phonemes with a stress digit count as CMU overrides.
    assert prep("Answer [A] is right.") == "Answer [A] is right."


def test_hindi_danda_not_followed_by_period():
    assert prep("नमस्ते! यह परीक्षण है।") == "नमस्ते! यह परीक्षण है।"
    assert prep("यह ठीक है॥") == "यह ठीक है॥"


def test_terminal_punctuation_added():
    assert prep("Hello world") == "Hello world."


def test_empty_text_passthrough():
    assert prep("") == ""
    assert prep("   **  ** ").strip(" .") == ""


if __name__ == "__main__":
    from tests.runner import run_tests

    run_tests(globals())
