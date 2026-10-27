# \# PI² — Microstructure \& Mémoire Longue (Données LOBSTER)

# 

# Projet d'Innovation Industrielle (PI²) — ESILV, Financial Engineering

# Encadrant : Jiang Pu (DVRC) · Sept. 2026 – Avr. 2027

# 

# \## Équipe

# \- Hamza Asadullah

# \- Baptiste Hétier

# \- Ilyas Kabchi

# \- Jules Matignon

# \- Baptiste Palasi

# \- Clément Romano

# \- Adli Bouzguenda

# 

# \## Objectif

# Analyser l'autocorrélation et les propriétés de mémoire longue de séries de prix

# haute fréquence (données LOBSTER), estimer l'exposant de Hurst (H) via la méthode

# des variations discrètes du fBm (Coeurjolly, 2000), et développer un moteur de

# backtesting Python pour valider des stratégies de trading.

# 

# \## Structure du repo

# data/

# &#x20; raw/              # données LOBSTER brutes (non versionnées, voir .gitignore)

# src/

# &#x20; hurst/            # estimateurs de H (scale connue / inconnue, filtres (-1,1) et (-1,2,-1))

# &#x20; microstructure/   # spread, order book imbalance, order flow

# &#x20; backtest/         # moteur de backtest et simulation d'ordres

# notebooks/          # exploration, visualisation (ACF/PACF, résultats H)

# tests/              # tests unitaires



# \## Installation

# ```bash

# pip install -r requirements.txt

# ```

# 

# \## Références

# \- Coeurjolly, J-F. (2000). \*Simulation and identification of the fractional

# &#x20; Brownian motion: a bibliographical and comparative study.\* JSS.

# \- Cont, R. (2005). \*Long range dependence in financial markets.\*

# \- Fama, E.F. \& French, K.R. (1988). \*Permanent and temporary components of

# &#x20; stock prices.\* JPE.

# \- Lo, A.W. (1991). \*Long-term memory in stock market prices.\* Econometrica.









