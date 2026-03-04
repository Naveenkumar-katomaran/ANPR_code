import re

def basic_plate_sanity(text: str) -> bool:
    """
    Fast rejection filter BEFORE postprocess.
    Prevent garbage entering sessions.
    """

    if not text:
        return False

    text = text.upper().replace(" ", "").replace("-", "")

    # too short / too long → reject
    if len(text) < 7 or len(text) > 12:
        return False

    # must start with 2 letters (Indian plates)
    if not re.match(r'^[A-Z]{2}', text):
        return False

    # must contain numbers
    if not any(c.isdigit() for c in text):
        return False

    return True