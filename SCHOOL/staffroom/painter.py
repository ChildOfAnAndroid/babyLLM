# CHARIS CAT 2025
# --- ʕっʘ‿ʘʔっ --- 
# BABYLLM // school/staffroom/painter.py
# v1.4


# painter.py
# ANSI terminal styling

class terminalFormat:
    RESET = "\033[0m"        # Normal terminal
    BOLD = "\033[1m"
    DIM = "\033[2m"          # Dim text colour
    UNDERLINE = "\033[4m"
    FLASH = "\033[5m"
    ITALIC = "\033[3m"

class terminalColour:
    # colours
    PURPLE = "\033[94m"
    MAGENTA = "\033[35m"
    BLUE = "\033[34m"
    GOLD = "\033[93m"
    RED_ALT = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    # 256 colour palette
    PALE_PURPLE = "\033[38;5;225m"
    ORANGE = "\033[38;5;52m"
    RED = "\033[38;5;124m"
    BRIGHT_RED = "\033[91m"

    # scales
    redBlue = {
        'brightRed': "\033[38;5;196m",
        'red': "\033[38;5;161m",
        'redPurple': "\033[38;5;126m",
        'purpleRed': "\033[38;5;91m",
        'purpleBlue': "\033[38;5;56m"
    }

    bluePink = {
        'blue': "\033[38;5;21m",
        'bluePurple': "\033[38;5;57m",
        'purple': "\033[38;5;93m",
        'purplePink': "\033[38;5;129m",
        'pink': "\033[38;5;165m",
        'brightPink': "\033[38;5;201m"
    }

    gradient = {
    'A': "\033[38;5;196m",
    'B': "\033[38;5;160m",
    'C': "\033[38;5;161m",
    'D': "\033[38;5;162m",
    'E': "\033[38;5;163m",
    'F': "\033[38;5;164m",
    'G': "\033[38;5;127m",
    'H': "\033[38;5;134m",
    'I': "\033[38;5;135m",
    'J': "\033[38;5;99m",
    'K': "\033[38;5;63m",
    'L': "\033[38;5;27m",
    'M': "\033[38;5;33m",
    'N': "\033[38;5;39m"
    }
