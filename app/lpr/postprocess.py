import re
from collections import Counter
from typing import List, Optional, Tuple, Dict, Any

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
# Normalize OCR text
# -------------------------------------------------
def normalize(text: str) -> str:
    if not text:
        return ""
    return text.upper().replace(" ", "").replace("-", "")


# -------------------------------------------------
# Character confusion fixes
# -------------------------------------------------
LETTER_FIX = {
    '0': 'O', '1': 'I', '2': 'Z', '5': 'S', '8': 'B', '6': 'G', '3': 'B'
}

NUMBER_FIX = {
    'O': '0', 'D': '0', 'I': '1', 'Z': '2', 'S': '5', 'B': '8', 'G': '6', 'T': '7'
}


# -------------------------------------------------
# Position-based correction (Anchor Logic)
# -------------------------------------------------
def correct_by_pattern(text: str) -> List[str]:
    """
    Align noisy text to an 11-char sparse template.
    Template: [0,1]=State, [2,3]=District, [4,5,6]=Series, [7,8,9,10]=UniqueID
    Returns exactly 11 elements (char or EMPTY_TOKEN).
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
            # txt_idx is L-1, L-2, L-3, L-4 (from end)
            # template_idx is 10, 9, 8, 7 (from end)
            txt_idx = L - 1 - i
            template_idx = 10 - i
            
            if txt_idx < 4: break # Safety: don't overlap with front anchor
            
            char = text[txt_idx]
            f = NUMBER_FIX.get(char, char)
            aligned[template_idx] = f if (isinstance(f, str) and f.isdigit()) else '?'

    # Anchor 3: MIDDLE (The 'float' part) -> Series LLL
    if L > 4:
        # middle start is index 4 in text
        # middle end is L-4 in text
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


def choose_best_plate(candidates: List[dict]) -> Tuple[Optional[str], float]:
    """
    ANCHOR-BASED SPARSE VOTING with SAFE RECONSTRUCTION.
    Preserves positional meaning and enforces final structural validation.
    """
    if not candidates:
        return None, 0.0

    # 1. Align candidates to sparse template
    voter_data: List[Dict[str, Any]] = []
    for c in candidates:
        raw_text = str(c.get("text", ""))
        conf_val = float(c.get("conf", 0.0))
        
        # Sliding window to find best alignment list of length 11
        aligned_list = extract_best_aligned_list(raw_text)
        if aligned_list:
            voter_data.append({"list": aligned_list, "conf": conf_val})

    if not voter_data:
        return None, 0.0

    # 2. Position-wise Voting (11 slots)
    reconstruction = [EMPTY_TOKEN] * 11
    total_batch_score = 0.0
    
    for pos in range(11):
        pos_votes: Dict[str, float] = {}
        for v in voter_data:
            char = v["list"][pos]
            weight = v["conf"]
            
            # Penalize '?' in results heavily
            if char == '?': weight *= 0.01
            
            # Boost weight for valid state codes
            if pos == 0 or pos == 1:
                st = "".join([str(v["list"][0]), str(v["list"][1])])
                if EMPTY_TOKEN not in st and '?' not in st:
                    if st in STATE_CODES: weight *= 4.0

            pos_votes[char] = pos_votes.get(char, 0.0) + weight

        if pos_votes:
            winner = max(pos_votes, key=lambda k: pos_votes[k])
            reconstruction[pos] = winner
            total_batch_score += float(pos_votes[winner])

    # 3. Safe Assembly (Structure-aware)
    # Filter out EMPTY but KEEP '?' for validation
    assembled_chars = [c for c in reconstruction if c != EMPTY_TOKEN]
    raw_plate = "".join(assembled_chars)
    
    # Base confidence
    avg_conf = total_batch_score / 11.0

    # 4. Final Validation & Repair
    # If the plate has '?', it's likely invalid, but we check if removing '?' 
    # creates a perfectly valid Indian plate format.
    final_plate = raw_plate.replace('?', '')
    
    if is_valid_format(final_plate):
        # Length Bias
        L = len(final_plate)
        if L == 10: avg_conf *= 1.2
        elif L == 9: avg_conf *= 1.05
        
        # Penalize if it had too many '?' originally
        q_count = raw_plate.count('?')
        if q_count > 0:
            avg_conf *= (1.0 - (q_count * 0.15))
            
        return final_plate, avg_conf

    # Rejection: If it doesn't match the regex, it's garbage
    return None, 0.0


def extract_best_aligned_list(text: str) -> Optional[List[str]]:
    """
    Sliding window to find the best 11-char sparse list representation.
    """
    text = normalize(text)
    if len(text) < 7: return None
    
    best_list = None
    max_score = -1000.0
    
    for length in range(7, 13):
        for i in range(len(text) - length + 1):
            sub = text[i:i+length]
            aligned = correct_by_pattern(sub)
            score = structure_score(aligned)
            
            # Massive bonus for standard 10-character plates
            actual_count = len([c for c in aligned if c != EMPTY_TOKEN and c != '?'])
            if actual_count == 10: score += 20.0
            
            if score > max_score:
                max_score = score
                best_list = aligned
                
    return best_list


def structure_score(aligned: List[str]) -> float:
    """
    Scores an 11-char alignment for quality.
    """
    if len(aligned) != 11: return -100.0
    
    s = 0.0
    for i, ch in enumerate(aligned):
        if ch == EMPTY_TOKEN:
            # Penalize empty slots in anchor zones
            if i in [0, 1, 2, 3, 7, 8, 9, 10]: s -= 1.0
            continue
        
        if ch == '?':
            s -= 5.0 # Very strong penalty for unknowns in important zones
            continue

        if i in [0, 1]: # State (L)
            s += 2.0 if (isinstance(ch, str) and ch.isalpha()) else -8.0
            if i == 1:
                state_pair = "".join([str(aligned[0]), str(aligned[1])])
                if state_pair in STATE_CODES: s += 25.0
        elif i in [2, 3]: # District (N)
            s += 2.0 if (isinstance(ch, str) and ch.isdigit()) else -8.0
        elif i in [4, 5, 6]: # Series (L)
            s += 1.0 if (isinstance(ch, str) and ch.isalpha()) else -3.0
        elif i in [7, 8, 9, 10]: # UniqueID (N)
            s += 2.5 if (isinstance(ch, str) and ch.isdigit()) else -8.0
            
    return s