"""SIM — Source credibility, in one place.

There were two of these. pass_d_score carried 40 outlets and weighted alert confidence
with them; forecast_engine carried 7 and weighted the weekly tension index. Anything
outside its own short list fell to forecast_engine's 0.6 default, so defense.gov,
aljazeera.com, wsj.com and nytimes.com — all scored 0.90-0.95 when deciding whether to
page — were treated as barely-above-unknown when deciding whether a country's risk was
rising. Two answers to one question, and the smaller list silently governed the trend.

The merge takes pass_d's table as the base, since it is the maintained one, and adds
the two wire services that only forecast_engine knew about. Where they disagreed
(bbc.com/bbc.co.uk at 1.0 vs 0.95) the lower value wins: pass_d places Reuters alone at
1.0 and that ordering is deliberate.
"""

# Domain -> credibility multiplier (0.0-1.0). Registrable domains only; subdomains
# resolve through the parent fallback in get_source_credibility.
SOURCE_CREDIBILITY: dict[str, float] = {
    # Wire services
    "reuters.com": 1.0,
    "apnews.com": 1.0,
    "afp.com": 1.0,
    # Broadcast / major press
    "bbc.co.uk": 0.95,
    "bbc.com": 0.95,
    "cnn.com": 0.90,
    "foxnews.com": 0.90,
    "wsj.com": 0.95,
    "nytimes.com": 0.95,
    "theguardian.com": 0.90,
    "france24.com": 0.90,
    "aljazeera.com": 0.90,
    # Official
    "defense.gov": 0.95,
    "centcom.mil": 0.95,
    "un.org": 0.95,
    "travel.state.gov": 0.95,
    # Regional press
    "timesofisrael.com": 0.95,
    "jpost.com": 0.95,
    "haaretz.com": 0.95,
    "ynetnews.com": 0.95,
    "presstv.ir": 0.85,
    "themoscowtimes.com": 0.85,
    "meduza.io": 0.85,
    "ukrinform.net": 0.90,
    "kyivindependent.com": 0.90,
    # Defence / security trade press
    "breakingdefense.com": 0.90,
    "militarytimes.com": 0.90,
    "warontherocks.com": 0.90,
    "longwarjournal.org": 0.90,
    "defenseone.com": 0.90,
    "defensenews.com": 0.90,
    "twz.com": 0.85,
    "al-monitor.com": 0.85,
    "dropsitenews.com": 0.85,
    # Research institutes
    "crisisgroup.org": 0.92,
    "bellingcat.com": 0.90,
    "thecipherbrief.com": 0.88,
    "foreignpolicy.com": 0.90,
    "warsawinstitute.org": 0.82,
    "jamestown.org": 0.88,
    "thesoufancenter.org": 0.88,
    "ctc.westpoint.edu": 0.92,
    "counterextremism.com": 0.85,
    # Social / aggregator
    "nitter.net": 0.80,
    "nitter.privacydev.net": 0.80,
    "nitter.poast.org": 0.80,
    "reddit.com": 0.50,
}

# What an outlet we have never rated is worth. Deliberately below every rated source:
# an unknown domain is not evidence of quality in either direction, and the corpus is
# long-tailed enough that most events arrive from one.
DEFAULT_CREDIBILITY = 0.6


def get_source_credibility(domain: str | None) -> float:
    """Credibility for a domain, falling back to its registrable parent.

    The two former implementations differed here as well: one walked the last two
    labels, the other suffix-matched every key. Suffix matching is kept because it
    handles multi-label public suffixes (bbc.co.uk) that a two-label split gets wrong.
    """
    if not domain:
        return DEFAULT_CREDIBILITY
    domain = domain.lower().strip()
    if domain in SOURCE_CREDIBILITY:
        return SOURCE_CREDIBILITY[domain]
    for parent, score in SOURCE_CREDIBILITY.items():
        if domain.endswith("." + parent):
            return score
    return DEFAULT_CREDIBILITY
