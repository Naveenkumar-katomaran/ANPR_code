import re
import logging
from collections import Counter
from typing import List, Optional, Tuple, Dict, Any
from app.config import *

# -------------------------------------------------
# Indian plate format
# Example: TN38A1234 (9), TN38BF7777 (10), TN38ABC1234 (11)
# General Structure: LL NN L{1,3} NNNN
# -------------------------------------------------
INDIAN_PATTERN = r'^[A-Z]{2}[0-9]{2}[A-Z]{1,3}[0-9]{4}$'

STATE_CODES = {
    'AN', 'AP', 'AR', 'AS', 'BR', 'CH', 'CG', 'DD', 'DL', 'GA', 'GJ', 'HR', 'HP', 'JK', 'JH', 'KA', 'KL', 'LA', 'LD', 'MP', 'MH', 'MN', 'ML', 'MZ', 'NL', 'OD', 'PY', 'PB', 'RJ', 'SK', 'TN', 'TS', 'TR', 'UP', 'UK', 'UA', 'WB'
}

EMPTY_TOKEN = "<EMPTY>"

# -------------------------------------------------
# 1. Normalization
# -------------------------------------------------
def normalize(text: str) -> str:
    if not text:
        return ""
    # uppercase, remove spaces and hyphens
    return text.upper().replace(" ", "").replace("-", "")


# -------------------------------------------------
# 2. Character Confusion Fixes
# -------------------------------------------------
LETTER_FIX = {
    '0': 'O', '1': 'I', '2': 'Z', '5': 'S', '8': 'B', '6': 'G', '3': 'B'
}

NUMBER_FIX = {
    'O': '0', 'D': '0', 'I': '1', 'Z': '2', 'S': '5', 'B': '8', 'G': '6', 'T': '7'
}


# -------------------------------------------------
# 3. Position-Based Correction (Anchor Logic)
# -------------------------------------------------
def correct_by_pattern(text: str) -> List[str]:
    """
    Align noisy text to an 11-char sparse template.
    Template: [0,1]=State, [2,3]=District, [4,5,6]=Series, [7,8,9,10]=UniqueID
    """
    text = normalize(text)
    if not text: return [EMPTY_TOKEN] * 11

    L = len(text)
    aligned: List[str] = [EMPTY_TOKEN] * 11
    
    # Anchor 1: FRONT (First 4) -> LL NN
    for i in range(min(4, L)):
        char = text[i]
        if i < 2: # State (Letters)
            f = LETTER_FIX.get(char, char)
            aligned[i] = f if (isinstance(f, str) and f.isalpha()) else '?'
        else: # District (Digits)
            f = NUMBER_FIX.get(char, char)
            aligned[i] = f if (isinstance(f, str) and f.isdigit()) else '?'

    # Anchor 2: BACK (Last 4) -> NNNN
    if L >= 4:
        for i in range(4):
            # txt_idx is L-1, L-2... template_idx is 10, 9...
            txt_idx = L - 1 - i
            template_idx = 10 - i
            if txt_idx < 4: break # Don't overlap starting 4
            
            char = text[txt_idx]
            f = NUMBER_FIX.get(char, char)
            aligned[template_idx] = f if (isinstance(f, str) and f.isdigit()) else '?'

    # Anchor 3: MIDDLE (The 'float' part) -> Series LLL
    if L > 4:
        # middle start is index 4, end is L-4
        mid_end = max(4, L - 4)
        mid_content = text[4:mid_end]
        for i, char in enumerate(mid_content):
            if i < 3: # Series slots 4, 5, 6
                f = LETTER_FIX.get(char, char)
                aligned[4+i] = f if (isinstance(f, str) and f.isalpha()) else '?'

    return aligned


def is_valid_format(text: str) -> bool:
    """Check if the text follows the LLNNL{1,3}NNNN pattern."""
    return bool(re.match(INDIAN_PATTERN, text))


# -------------------------------------------------
# 4. Sparse Voting (Temporal Aggregation)
# -------------------------------------------------
def choose_best_plate(candidates: List[dict]) -> Tuple[Optional[str], float]:
    """
    Aggregates multiple OCR frames using positional voting and structure scoring.
    """
    if not candidates:
        return None, 0.0

    # Align each candidate to the 11-slot template
    voter_data = []
    for c in candidates:
        raw_text = str(c.get("text", ""))
        conf_val = float(c.get("conf", 0.0))
        
        # Try different alignments? (Simplified: strictly correct_by_pattern)
        # We use a sliding window logic to find the best sub-string to align
        clist = extract_best_aligned_list(raw_text)
        if clist:
            voter_data.append({"list": clist, "conf": conf_val})

    if not voter_data:
        return None, 0.0

    # 11-slot voting
    reconstruction = [EMPTY_TOKEN] * 11
    total_weights = 0.0
    
    for pos in range(11):
        pos_votes: Dict[str, float] = {}
        for ev in voter_data:
            char = ev["list"][pos]
            weight = ev["conf"]
            
            # Penalize unknowns (?) heavily
            if char == '?': weight *= 0.01
            
            # Special Rule: Give REAL CHARACTERS a presence weight boost
            # to prevent 'EMPTY_TOKEN' from winning just because 
            # some frames missed the character or misaligned.
            if char != EMPTY_TOKEN and char != '?':
                weight *= 1.5 
            
            # State code boost (TN, KA, etc)
            if pos == 0 or pos == 1:
                s0 = ev["list"][0]
                s1 = ev["list"][1]
                if s0 != EMPTY_TOKEN and s1 != EMPTY_TOKEN and s0 != '?' and s1 != '?':
                    st = "".join([get_char_safe(s0), get_char_safe(s1)])
                    if st in STATE_CODES: weight *= 4.0

            pos_votes[char] = pos_votes.get(char, 0.0) + weight

        if pos_votes:
            # Check if any character is close to 'EMPTY' 
            # If we have real character votes, we prefer them if they are common enough
            winner = max(pos_votes, key=lambda k: pos_votes[k])
            
            # If EMPTY wins but there's a strong character runner-up, 
            # we check if that runner-up appears in a significant % of frames.
            if winner == EMPTY_TOKEN:
                eligible_chars = {k: v for k, v in pos_votes.items() if k != EMPTY_TOKEN and k != '?'}
                if eligible_chars:
                    best_char = max(eligible_chars, key=lambda k: eligible_chars[k])
                    # If best char has at least 30% of the total POS weight, pick it
                    total_pos_weight = sum(pos_votes.values())
                    if eligible_chars[best_char] / total_pos_weight > 0.3:
                        winner = best_char
                        
            reconstruction[pos] = winner
            total_weights += pos_votes[winner]

    # Assembled string
    raw_plate = "".join([c for c in reconstruction if c != EMPTY_TOKEN])
    
    # 5. Final repair & validation
    final_plate = raw_plate.replace('?', '')
    
    if is_valid_format(final_plate) and VALIDATE_INDIAN_PLATE:
        # 6. Confidence Scoring
        avg_conf = total_weights / 11.0
        
        # Penalties/Bonuses
        L = len(final_plate)
        if L == 10: avg_conf *= 1.2
        elif L == 9: avg_conf *= 1.05
        
        q_count = raw_plate.count('?')
        if q_count > 0:
            avg_conf *= (1.0 - (q_count * 0.15))
            
        return final_plate, avg_conf
    else:
        avg_conf = total_weights / 11.0
        return final_plate, avg_conf

    logging.info(f"[POSTPROCESS] Rejected: raw={raw_plate} final={final_plate} valid={is_valid_format(final_plate)}")
    return None, 0.0


def extract_best_aligned_list(text: str) -> Optional[List[str]]:
    """
    Sliding window to find the sub-string of text (lengths 7-12)
    that aligns most cleanly to our pattern.
    """
    text = normalize(text)
    if len(text) < 7: return None
    
    best_list = None
    max_score = -9999.0
    
    # Try different offsets and lengths
    for length in range(7, 13):
        for i in range(len(text) - length + 1):
            sub = text[i:i+length]
            aligned = correct_by_pattern(sub)
            score = structure_score(aligned)
            
            # Standard 10-char bias
            actual_count = len([c for c in aligned if c != EMPTY_TOKEN and c != '?'])
            if actual_count == 10: score += 15.0
            
            if score > max_score:
                max_score = score
                best_list = aligned
                
    return best_list


# -------------------------------------------------
# 5. Structure Scoring
# -------------------------------------------------
def structure_score(aligned: List[str]) -> float:
    """
    Evaluates how well an 11-slot alignment matches Indian plate rules.
    """
    if len(aligned) != 11: return -100.0
    
    s = 0.0
    for i, ch in enumerate(aligned):
        if ch == EMPTY_TOKEN:
            # Penalize empty in mandatory slots
            if i in [0, 1, 2, 3, 7, 8, 9, 10]: s -= 1.0
            continue
        
        if ch == '?':
            s -= 5.0 # Heavy penalty for logic-defying characters
            continue

        # LL NN LLL NNNN
        if i in [0, 1]: # State (L)
            s += 2.0 if ch.isalpha() else -8.0
            if i == 1:
                st = "".join([get_c(aligned[0]), get_c(aligned[1])])
                if st in STATE_CODES: s += 25.0
        elif i in [2, 3]: # Dist (N)
            s += 2.0 if ch.isdigit() else -8.0
        elif i in [4, 5, 6]: # Series (L)
            s += 1.0 if ch.isalpha() else -3.0
        elif i in [7, 8, 9, 10]: # Unique (N)
            s += 2.5 if ch.isdigit() else -10.0
            
    return s

def get_char_safe(v):
    return str(v) if (v != EMPTY_TOKEN and v != '?') else ""

def get_c(v):
    return str(v) if (v != EMPTY_TOKEN and v != '?') else ""



