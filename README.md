# Explainable AI-Based Phishing URL Detection & Cybersecurity Analysis Platform

Defensive-security academic project (MCA minor project). The platform
detects phishing URLs from lexical and statistical URL features using
machine learning, explains its decisions with SHAP, extracts IOCs, keeps
a scan history, and presents results in a SOC-style investigation console.

**This is a defensive analysis tool. It never visits, renders, or hosts
malicious content. It only analyzes URL strings and public metadata.**

## Current status

- Phase 0 — project foundation (structure + virtual environment): complete.
- Upcoming phases: dataset collection → feature engineering → model
  training → calibration & thresholding → SHAP explainability → Flask
  backend → SOC frontend → testing & documentation.

This README is intentionally minimal at this stage. Full installation,
dataset, methodology, and viva-oriented documentation will be written in
the final phase, once the system demonstrably works end to end.

## Quick start (Fedora Workstation)

    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install --upgrade pip
    python -c "import dataset, features, model, services; print('OK')"

## Structure

    app.py            Flask application entry point (later phase)
    config.yaml       Application configuration (no secrets)
    dataset/          Dataset acquisition and preprocessing code
    features/         Feature engineering (lexical, entropy, WHOIS, DNS, IOC)
    model/            Training code + persisted model artifacts
    services/         Prediction and SHAP explanation services
    templates/        SOC-style frontend (Flask/Jinja2)
    static/           Frontend CSS/JS
    data/             Raw feeds, processed dataset, SQLite history
    tests/            pytest test suite
    logs/             Rotating application logs
