"""Apply source patches by tokens so harmless formatting cannot disable a patch.

Comments and whitespace are ignored for matching; string contents and operator
characters are retained. Callers still require the expected match count before
patching. This is intended for the pinned C/GLSL/Python build adapters, not as a
general-purpose source refactoring parser.
"""

import ast
import re

_TOKENS = re.compile(
    r"//[^\n]*|/\*[\s\S]*?\*/|#[^\n]*$|"
    r'"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'|'
    r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|'
    r"[A-Za-z_]\w*|\d+(?:\.\d*)?(?:[eE][+-]?\d+)?[uUlLfF]*|[^\s]",
    re.MULTILINE,
)


_PYTHON_TOKENS = re.compile(
    r'#[^\n]*|"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'|'
    r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|'
    r"[A-Za-z_]\w*|\d+(?:\.\d*)?(?:[eE][+-]?\d+)?[uUlLfF]*|[^\s]",
    re.MULTILINE,
)


def _tokens(text, python_source=False):
    tokens = []
    for match in (_PYTHON_TOKENS if python_source else _TOKENS).finditer(text):
        value = match[0]
        if (python_source and value.startswith("#")) or (
            not python_source and value.startswith(("//", "/*"))
        ):
            continue
        # Preserve preprocessor lines as tokens; Python comments need not match
        # any of our pinned code anchors. A final comma before a closing bracket
        # may be inserted by a formatter without changing the patched call.
        tokens.append((value, match.start(), match.end()))
    return [
        token
        for index, token in enumerate(tokens)
        if not (token[0] == "," and index + 1 < len(tokens) and tokens[index + 1][0] in ")]}")
    ]


def _matches(text, anchor):
    # Python's // is floor division, not a C comment. These adapters patch whole
    # valid modules, so parsing the source reliably selects its lexical rules.
    try:
        ast.parse(text)
        python_source = True
    except (SyntaxError, ValueError):
        python_source = False
    haystack = _tokens(text, python_source)
    needle = [token[0] for token in _tokens(anchor, python_source)]
    if not needle:
        raise ValueError("A source patch needs a nonempty code anchor")
    width = len(needle)
    first = needle[0]
    found = []
    for index, token in enumerate(haystack):
        if token[0] == first and [part[0] for part in haystack[index : index + width]] == needle:
            found.append((token[1], haystack[index + width - 1][2]))
    return found


def code_count(text, anchor):
    """Count code occurrences while excluding comment/string lookalikes."""
    return len(_matches(text, anchor))


def replace_code(text, anchor, replacement, expected=None):
    """Replace complete token matches; reject missing/ambiguous required patches."""
    matches = _matches(text, anchor)
    if expected is not None and len(matches) != expected:
        raise RuntimeError(
            f"Source patch expected {expected} match(es), got {len(matches)}: {anchor[:100]!r}"
        )
    # Matching begins at the first token. Keep its existing line indentation
    # rather than adding the anchor/replacement indentation a second time.
    replacement = replacement.lstrip(" \t")
    for begin, end in reversed(matches):
        text = text[:begin] + replacement + text[end:]
    return text
