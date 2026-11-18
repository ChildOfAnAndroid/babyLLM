"""Shared text preparation and rendering for chat and training."""

import re


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


_AMERICAN_TO_BRITISH = {
    "analyze": "analyse",
    "analyzed": "analysed",
    "analyzes": "analyses",
    "analyzing": "analysing",
    "apologize": "apologise",
    "apologized": "apologised",
    "apologizes": "apologises",
    "apologizing": "apologising",
    "authorize": "authorise",
    "authorized": "authorised",
    "authorizes": "authorises",
    "authorizing": "authorising",
    "authorization": "authorisation",
    "authorizations": "authorisations",
    "behavior": "behaviour",
    "behaviors": "behaviours",
    "center": "centre",
    "centers": "centres",
    "centered": "centred",
    "centering": "centring",
    "color": "colour",
    "colors": "colours",
    "colored": "coloured",
    "coloring": "colouring",
    "customize": "customise",
    "customized": "customised",
    "customizes": "customises",
    "customizing": "customising",
    "defense": "defence",
    "offense": "offence",
    "offenses": "offences",
    "dialog": "dialogue",
    "dialogs": "dialogues",
    "favorite": "favourite",
    "favorites": "favourites",
    "favorited": "favourited",
    "favoriting": "favouriting",
    "favor": "favour",
    "favors": "favours",
    "favored": "favoured",
    "favoring": "favouring",
    "gray": "grey",
    "humor": "humour",
    "humors": "humours",
    "honor": "honour",
    "honors": "honours",
    "honored": "honoured",
    "honoring": "honouring",
    "honorable": "honourable",
    "honorably": "honourably",
    "labor": "labour",
    "labors": "labours",
    "labored": "laboured",
    "laboring": "labouring",
    "license": "licence",
    "licenses": "licences",
    "licensing": "licencing",
    "neighbor": "neighbour",
    "neighbors": "neighbours",
    "neighboring": "neighbouring",
    "organize": "organise",
    "organized": "organised",
    "organizes": "organises",
    "organizing": "organising",
    "organization": "organisation",
    "organizations": "organisations",
    "personalize": "personalise",
    "personalized": "personalised",
    "personalizes": "personalises",
    "personalizing": "personalising",
    "realize": "realise",
    "realized": "realised",
    "realizes": "realises",
    "realizing": "realising",
    "recognize": "recognise",
    "recognized": "recognised",
    "recognizes": "recognises",
    "recognizing": "recognising",
    "summarize": "summarise",
    "summarized": "summarised",
    "summarizes": "summarises",
    "summarizing": "summarising",
    "synchronize": "synchronise",
    "synchronized": "synchronised",
    "synchronizes": "synchronises",
    "synchronizing": "synchronising",
    "synchronization": "synchronisation",
    "theater": "theatre",
    "theaters": "theatres",
    "rumor": "rumour",
    "rumors": "rumours",
    "savor": "savour",
    "savors": "savours",
    "savored": "savoured",
    "savoring": "savouring",
    "traveled": "travelled",
    "traveling": "travelling",
    "traveler": "traveller",
    "travelers": "travellers",
}

_AMERICAN_TO_BRITISH_RE = re.compile(
    r"\b("
    + "|".join(
        sorted(map(re.escape, _AMERICAN_TO_BRITISH.keys()), key=len, reverse=True)
    )
    + r")\b",
    flags=re.IGNORECASE,
)


def _match_word_case(source: str, replacement: str) -> str:
    if source.isupper():
        return replacement.upper()
    if source[:1].isupper():
        return replacement.capitalize()
    return replacement


def to_british_english(text: str) -> str:
    """Convert common US spellings in *text* to British spellings."""
    if not isinstance(text, str) or not text:
        return text

    def _replace(match: re.Match) -> str:
        source = match.group(0)
        replacement = _AMERICAN_TO_BRITISH.get(source.lower(), source)
        return _match_word_case(source, replacement)

    return _AMERICAN_TO_BRITISH_RE.sub(_replace, text)
