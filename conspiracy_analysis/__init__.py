"""
conspiracy_analysis: A modular package for analyzing COVID-19 conspiracy theory
spreading dynamics on social networks.

Core components:
- data: Loading the public anonymized graph and preprocessing event histories
- models: Cox proportional hazards (first-time adoption), Hawkes process (repeated sharing),
          baseline hazard parametrization (linear & Weibull)
- analysis: Semantic clustering with Silhouette Score optimization, barrier analysis
- simulation: Agent-based model combining Cox + Hawkes on hourly time steps
- visualization: Publication-quality figures
"""

__version__ = "1.0.0"

# Temporal constants used throughout the package
HOUR_RESOLUTION = 1       # hours per discrete time step
EXPOSURE_WINDOW = int(14 * 24)  # 14 days or 336 h for rolling neighbor exposure
CONSPIRACY_PROB_THRESHOLD = 0.8  # minimum probability to classify a tweet as conspiracy
BOT_SCORE_THRESHOLD = 0.4  # threshold for bot vs. human classification
SIMULTANEOUS_NUDGE = (1/60)    # hours to shift time forward when entry == time
