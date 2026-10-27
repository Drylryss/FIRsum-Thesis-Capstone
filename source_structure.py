"""Shared conservative boundaries for a decision and appended separate opinions."""
import re

# Require an uppercase heading, not a reference to a dissent in ordinary prose.
OPINION_HEADING = re.compile(
    r'\b(?:(?:SEPARATE\s+)?(?:CONCURRING(?:\s+AND\s+DISSENTING)?|'
    r'DISSENTING(?:\s+AND\s+CONCURRING)?)|SEPARATE)\s+OPINION\b')


def before_separate_opinion(text):
    match = OPINION_HEADING.search(text)
    return match.start() if match else len(text)
