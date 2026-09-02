"""
AegisAI — URL Feature Extraction

Extracts engineered features from raw URLs for the phishing classifier.
Features include structural properties (length, depth, special characters),
lexical properties (entropy, digit ratios), and domain-level signals.

Usage:
    from src.preprocessing.url_features import extract_url_features
    features = extract_url_features("https://example.com/login")
"""

from __future__ import annotations

import math
import re
from urllib.parse import urlparse
import pandas as pd


def _calculate_entropy(text: str) -> float:
    """Calculate the Shannon entropy of a string."""
    if not text:
        return 0.0
    entropy = 0.0
    for x in set(text):
        p_x = float(text.count(x)) / len(text)
        if p_x > 0:
            entropy += - p_x * math.log(p_x, 2)
    return entropy


def extract_url_features(url: str) -> dict[str, float | int]:
    """Extract a dictionary of engineered features from a single URL.

    Parameters
    ----------
    url : str
        The raw URL string to analyse.

    Returns
    -------
    dict[str, float | int]
        Feature name → value mapping.
    """
    if not isinstance(url, str):
        url = str(url)
    
    # Pre-parse URL
    # Add scheme if missing to help urlparse
    if not url.startswith(('http://', 'https://', 'ftp://')):
        parsed_url = urlparse('http://' + url)
    else:
        parsed_url = urlparse(url)

    hostname = parsed_url.hostname or ""
    path = parsed_url.path or ""
    query = parsed_url.query or ""
    
    # IP address detection
    ip_pattern = re.compile(
        r'^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}'
        r'(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$'
    )
    has_ip = 1 if ip_pattern.match(hostname) else 0

    # Subdomain count approximation (number of dots in hostname, minus 1 for TLD, bounded to 0)
    num_subdomains = max(0, hostname.count('.') - 1) if not has_ip else 0

    features = {
        "url_length": len(url),
        "hostname_length": len(hostname),
        "path_length": len(path),
        "query_length": len(query),
        "num_dots": url.count('.'),
        "num_hyphens": url.count('-'),
        "num_underscores": url.count('_'),
        "num_special_chars": len(re.findall(r'[^a-zA-Z0-9]', url)),
        "num_digits": len(re.findall(r'[0-9]', url)),
        "num_subdomains": num_subdomains,
        "has_https": 1 if parsed_url.scheme == 'https' else 0,
        "has_ip_address": has_ip,
        "has_at_symbol": 1 if '@' in url else 0,
        "url_entropy": _calculate_entropy(url),
        "domain_length": len(hostname), # Often similar to hostname_length
        "path_depth": path.count('/') if path else 0,
        "query_param_count": query.count('&') + 1 if query else 0
    }
    
    return features


def extract_url_features_batch(urls: pd.Series) -> pd.DataFrame:
    """Extract features for a batch of URLs.

    Parameters
    ----------
    urls : pd.Series
        Series of raw URL strings.

    Returns
    -------
    pd.DataFrame
        DataFrame where each row is a feature vector for one URL.
    """
    # Apply feature extraction to each URL
    features_list = urls.apply(extract_url_features)
    # Convert list of dicts to DataFrame
    return pd.DataFrame(features_list.tolist(), index=urls.index)
