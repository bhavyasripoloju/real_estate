# Hyderabad Real Estate EDA & Demo

This repository contains a small exploratory data analysis (EDA) and a Streamlit demo for a Hyderabad house price dataset.

Contents
- `hyderabad_eda.py` — cleaning script: finds the CSV, coerces numeric columns, drops NaNs, and saves cleaned outputs.
- `hyderabad_data_cleaned.csv` / `hyderabad_data_cleaned.pkl` — cleaned dataset produced by the script.
- `plot_mean_price_per_location.png`, `boxplot_price_by_location.png`, `scatter_area_vs_price.png` — saved visualizations.
- `hyd_plots.zip` — zip archive with the three generated plots.
- `streamlit_app.py` — small Streamlit demo app that loads the cleaned data and displays interactive charts.
- `requirements.txt` — Python dependencies for the demo.

Quick run (recommended: use a virtual environment)

```powershell
cd "C:\Users\admin\OneDrive\Desktop\Hyd real estate"
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
# create cleaned data (if not present) and generate plots
python hyderabad_eda.py --save-pickle
# run Streamlit demo
streamlit run streamlit_app.py
```

Notes
- The cleaning script drops rows with NaN values for core numeric fields (price(L), rate_persqft, area_insqft). If you prefer imputation instead of dropping, edit `hyderabad_eda.py`.
- The Streamlit app expects `hyderabad_data_cleaned.pkl` (or CSV) in the same directory.

Resume blurb (1–2 lines)
---------------------------------
Hyderabad real-estate EDA & demo: cleaned and analyzed a 3,600+ listing dataset; built interactive Streamlit dashboard with location-level price summaries, boxplots, and area-vs-price visualizations. (Python, pandas, matplotlib/seaborn, Streamlit)

License
-------
Add a license file if you want to publish this code publicly (e.g., `LICENSE` with MIT).
