"""
source_filters.py

Esclude le source con problemi noti di label quality dalle analisi.
"""

# BeaverTails: 6.96% errore di etichetta stimato (Zhu et al. 2024, "Unmasking
# and Improving Data Credibility"). Escluso sempre.
EXCLUDED_SOURCES = {"beavertails"}


def filter_sources(df, source_col: str = "source"):
    return df[~df[source_col].isin(EXCLUDED_SOURCES)].copy()
