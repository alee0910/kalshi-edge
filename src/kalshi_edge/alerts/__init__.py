"""Email alerts via Gmail SMTP with app-password auth.

Planned:
    templates/digest.html.j2
    templates/realtime.html.j2
    smtp.py       — send with TLS + attachment support
    digest.py     — build top-N daily email
    realtime.py   — per-contract alerts keyed on edge + model_confidence
"""
