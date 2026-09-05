"""Shannon entropy of character distributions.

H(X) = -sum( p(x) * log2(p(x)) ) over the characters of a string.

Entropy quantifies uncertainty/randomness in the character
distribution. 'example.com' reuses common characters and has lower
entropy; algorithmically generated hosts like 'xjkqp91kd.xyz' may show
higher entropy.

LIMITATION (documented, not hidden): high entropy is NOT proof of
phishing. Benign CDNs, IDN domains and long paths can be high-entropy.
This is one weak feature among many - never a decision rule.
"""

from __future__ import annotations

import math
from collections import Counter


def shannon_entropy(s: str) -> float:
    """Shannon entropy (bits/char) of a string. Empty string -> 0.0."""
    if not s:
        return 0.0
    counts = Counter(s)
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())
