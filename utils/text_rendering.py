"""Pure text rendering helpers shared by chat output and tutor displays."""


def decode_bpe_whitespace(text, *, markers="Ġ▁Ċĉ"):
    """Decode display whitespace markers without altering literal underscores."""
    whitespace = {"Ġ": " ", "▁": " ", "Ċ": "\n", "ĉ": "\t"}
    return str(text).translate(str.maketrans({key: whitespace[key] for key in markers}))


def render_token_text(text, *, token_view=False, replacements=()):
    """Render reserved tokens, optionally keeping whitespace visible in token bars."""
    rendered = "" if text is None else str(text)
    if token_view:
        rendered = rendered.replace("\r", "").replace("\n", "Ċ").replace("\t", "ĉ")
    for token, label in replacements:
        if token and label:
            rendered = rendered.replace(token, label)
    return rendered
