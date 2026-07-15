"""Shared colour palette for the MMH0 paper figures."""

# Base palette, used directly for categorical series (galaxies, recon
# variants).
PALETTE = ["#0091ad", "#f25f5c", "#813405", "#f0c808", "#64113f"]
TEAL, RED, BROWN, GOLD, PLUM = PALETTE

# Spectral classes: blueshifted (approaching) and redshifted (receding) spots.
BLUESHIFTED = TEAL
REDSHIFTED = RED

# H0 inference selection variants, with the redshift selection on the
# redder side.
SELECTION = {"none": PLUM, "distance": TEAL, "redshift": RED}

# Literature reference values.
PLANCK_C = BROWN
SHOES_C = GOLD
