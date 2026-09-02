# AegisAI — Data Directory

This directory holds all datasets used by AegisAI.

## Structure

```
data/
├── raw/          # Original, unmodified datasets
└── processed/    # Cleaned, split, and feature-engineered datasets
```

## Expected Datasets

The following datasets are required for the **data collection phase**. Please download them from their official sources and place them in the `data/raw/` directory.

### URL Datasets (Phishing & Legitimate)
| Dataset | Filename | Expected Columns | Source / Instructions |
|---|---|---|---|
| PhishTank | `phishtank.csv` | `url`, `label` | Download from [PhishTank](https://www.phishtank.com/). Registration required. (1=phishing) |
| OpenPhish | `openphish.csv` | `url`, `label` | Download from [OpenPhish](https://openphish.com/). Contains phishing URLs. |
| Kaggle URLs | `kaggle_phishing_urls.csv` | `url`, `label` | e.g. [Web Page Phishing Detection Dataset](https://www.kaggle.com/datasets/shashwatwork/web-page-phishing-detection-dataset). |

### Text Datasets (Spam & Ham)
| Dataset | Filename | Expected Columns | Source / Instructions |
|---|---|---|---|
| UCI SMS Spam | `uci_sms_spam.csv` | `v1` (label), `v2` (text) | Download from [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection). |
| SpamAssassin | `spamassassin_emails.csv` | `text`, `label` | Public corpus from Apache SpamAssassin. Requires parsing into CSV. |

## Conventions
- **Labels**: `0` indicates Legitimate/Ham. `1` indicates Phishing/Spam.
- Missing or malformed records will be automatically pruned during preprocessing.

## Licensing and Redistribution
> [!WARNING]
> Do NOT bypass licensing restrictions. Do NOT redistribute restricted datasets by committing them to Git. Keep all raw data local.

## Preprocessing Commands
Once you have placed the raw datasets in `data/raw/`, run the following commands to generate the cleaned datasets in `data/processed/`:

```bash
python scripts/prepare_url_data.py
python scripts/prepare_text_data.py
```

## Important
- **Do not commit raw data to Git.** The `data/` contents are gitignored.
- Preprocessing scripts will output to `data/processed/`.
