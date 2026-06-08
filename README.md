# Dota 2 Draft Analysis

Interactive draft simulator for Dota 2 patch 7.41, trained on ~18k high-MMR (Divine+) public matches from OpenDota. Click hero portraits or type to draft a 5v5 and the model returns a calibrated radiant win probability plus a phase-advantage curve across game length.

## Quick start

```bash
pip install -r requirements.txt
streamlit run src/dashboard.py
```

Then open <http://localhost:8501>. The bundled `data/matches.csv` (~2 MB) and `data/model.pkl` let the dashboard run out of the box.

## Deploy publicly (mobile-accessible)

The fastest path is **Streamlit Community Cloud** — free, designed for Streamlit, gives you a public `*.streamlit.app` URL that works on mobile:

1. Push this repo to your own GitHub if it isn't already.
2. Sign in at <https://share.streamlit.io> with GitHub.
3. **New app** → pick this repo → branch `main` → main file `src/dashboard.py`.
4. (Optional) **Settings → Secrets** → paste the contents of `.streamlit/secrets.toml.example` and fill in your real PandaScore key.
5. Deploy. The URL is shareable and mobile-friendly.

Note: Vercel is *not* a good target — it's built for stateless serverless functions, and Streamlit needs a long-running WebSocket connection. Hugging Face Spaces is a fine alternative if you prefer it over Streamlit Cloud.

## What's in here

| File | Purpose |
| --- | --- |
| `src/dashboard.py` | Streamlit dashboard — clickable portrait grid, type-to-add, win probability, phase chart |
| `src/predict.py` | `DotaModel` class: predict(), hero strengths, top synergies, top counters |
| `src/train_model.py` | Trains the sparse L2-regularized logistic regression on `data/matches.csv` |
| `src/fetch_matches.py` | Pulls fresh high-MMR public matches from OpenDota (resumable, rate-limited) |
| `src/sweep.py` | Feature-set + regularization sweep for picking C |
| `data/heroes.json` | Hero constants from OpenDota (id, internal name, attribute) |
| `data/matches.csv` | 18k drafted full-5v5 matches on patch 7.41 |
| `data/model.pkl` | Pickled trained model (LogisticRegression + feature offsets + hero mapping) |

## The model

L2-regularized binary logistic regression with sparse binary features:

```
P(radiant wins) = σ( b + Σᵢ βᵢ · xᵢ )         σ(z) = 1 / (1 + e^-z)
```

| feature group | size | fires when |
| --- | --- | --- |
| `hero_radiant[h]` | 127 | radiant picked hero h |
| `hero_dire[h]` | 127 | dire picked hero h |
| `synergy_radiant[h₁,h₂]` | 8,001 | both heroes on radiant |
| `synergy_dire[h₁,h₂]` | 8,001 | both on dire |
| `counter[hᵣ,h_d]` | 16,129 | radiant hᵣ vs dire h_d |
| **total** | **32,385** | **~55 fire per match** |

Trained with `C=0.005` (strong L2). Test AUC ≈ 0.557, log-loss 0.6858 vs baseline 0.6912.

"Solo strength" displayed in the dashboard = `(β_hero_radiant − β_hero_dire) / 2`, the side-independent log-odds contribution of picking that hero.

## Honest accuracy notes

- **AUC 0.557 is modest** — draft alone doesn't explain most of a Dota game. Industry tools peak around 0.60–0.62 trained on millions of matches.
- **Hero-strength signal is real**; pair/counter coefficients are near the noise floor at 18k matches (~25 obs per pair) and should be read as hints, not facts.
- **The phase-advantage curve has two known biases**: (1) game duration is endogenous — heroes that scale partly *cause* games to run long when they're winning; (2) it sums 5 individual hero winrates, ignoring teammate effects.

## Refreshing data

```bash
# Pull more Divine+ matches on patch 7.41 (resumes from existing CSV)
python src/fetch_matches.py --target 50000

# Retrain
python src/train_model.py --C 0.005

# Restart the dashboard
streamlit run src/dashboard.py
```

OpenDota's public API is rate-limited to 60 calls/min and 2,000/day without an API key. Each call yields ~100 matches, of which ~70% are kept (Divine+ ranked drafts on patch 7.41).

## Roadmap

- Pair-aware aggregate metrics in the result panel (cache is built, UI not yet wired)
- Phase-bucketed pair winrates (needs ~100k matches for stable per-pair-per-phase estimates)
- Pro match overlay (`/proMatches` + per-match picks, smaller volume but real team data)

## License

MIT.
