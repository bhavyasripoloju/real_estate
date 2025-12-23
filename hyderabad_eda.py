"""hyderabad_eda.py

Simple cleaning script for the Hyderabad house price dataset.

What it does:
- Locates a CSV in the script directory (tries common filenames and falls back to first CSV).
- Loads the CSV using pandas.
- Coerces common numeric columns to numbers.
- Drops any rows that still contain NaNs after coercion (removes incomplete rows).
- Drops an automatic index column named 'Unnamed: 0' if present.
- Saves cleaned data to `hyderabad_data_cleaned.csv` and `hyderabad_data_cleaned.pkl`.
- Prints before/after counts and basic summary.

Usage (PowerShell):
    python "hyderabad_eda.py"

This script is intentionally conservative: it prints what it changed and where it saved the cleaned files.
"""

import os
import sys
from typing import Optional

import argparse
import pandas as pd
import matplotlib
# Use Agg backend for environments without a display (prevents Tkinter backend errors)
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


def find_csv_file(script_directory: str) -> Optional[str]:
    candidate_names = [
        'Hyderbad_House_price.csv',
        'Hyderabad_House_price.csv',
        'hyderabad_House_price.csv',
        'Hyderbad_House_price.csv',
        'Hyderbad_House_price.CSV',
    ]

    for name in candidate_names:
        p = os.path.join(script_directory, name)
        if os.path.exists(p):
            return p

    # Fallback: first CSV in folder
    csv_files = [f for f in os.listdir(script_directory) if f.lower().endswith('.csv')]
    if csv_files:
        return os.path.join(script_directory, csv_files[0])
    return None


def coerce_numeric_columns(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    # Drop a potential auto-index column that pandas sometimes creates on save
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])

    # Columns we expect to be numeric in this dataset
    numeric_cols = ['price(L)', 'rate_persqft', 'area_insqft']
    df = coerce_numeric_columns(df, numeric_cols)

    # Remove rows with any NaNs (user requested to remove NaN values)
    before = len(df)
    df = df.dropna()
    after = len(df)

    print(f"Rows before dropna: {before}")
    print(f"Rows after dropna:  {after}")
    print(f"Removed rows:       {before - after}")

    # Reset index after dropping rows
    df = df.reset_index(drop=True)
    return df


def main():
    parser = argparse.ArgumentParser(description='Clean and optionally plot Hyderabad house price data')
    parser.add_argument('--plots', action='store_true', help='Generate and save three visualizations')
    args = parser.parse_args()

    script_directory = os.path.dirname(os.path.abspath(__file__))
    file_path = find_csv_file(script_directory)
    if file_path is None:
        print("❌ No CSV file found in the script directory. Place the dataset CSV next to this script.")
        sys.exit(1)

    print(f"Using data file: {file_path}")
    df = pd.read_csv(file_path)

    print("-" * 40)
    print(f"Loaded rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print("-" * 40)

    # Show missing values summary before cleaning
    missing_before = df.isna().sum()
    print('Missing values (before):')
    print(missing_before[missing_before > 0].to_string() if missing_before.any() else 'No missing values')

    # Clean
    df_clean = clean_dataframe(df)

    # Show missing values after cleaning (should be none)
    missing_after = df_clean.isna().sum()
    print('Missing values (after):')
    print(missing_after[missing_after > 0].to_string() if missing_after.any() else 'No missing values')

    # Save cleaned outputs
    cleaned_csv = os.path.join(script_directory, 'hyderabad_data_cleaned.csv')
    cleaned_pkl = os.path.join(script_directory, 'hyderabad_data_cleaned.pkl')

    df_clean.to_csv(cleaned_csv, index=False)
    df_clean.to_pickle(cleaned_pkl)

    print(f"Saved cleaned CSV to: {cleaned_csv}")
    print(f"Saved cleaned pickle to: {cleaned_pkl}")

    # Print a tiny summary of cleaned data
    print('-' * 40)
    print('Cleaned data preview:')
    with pd.option_context('display.max_rows', 5, 'display.max_columns', None):
        print(df_clean.head())

    if args.plots:
        generate_plots(df_clean, script_directory)


def generate_plots(df: pd.DataFrame, script_directory: str) -> None:
    """Create three visualizations and save them to the script directory.

    1) Bar plot of mean price per location (top 20 locations by count)
    2) Boxplot of price by location (top 10 locations by count)
    3) Scatter plot of area_insqft vs price(L) with regression line
    """
    sns.set(style='whitegrid')

    # Ensure numeric columns exist
    if 'price(L)' not in df.columns or 'area_insqft' not in df.columns:
        print('Required columns for plots not found: price(L), area_insqft')
        return

    # Top locations by count
    location_counts = df['location'].value_counts()
    top20 = location_counts.head(20).index.tolist()

    # 1) Mean price per location (for top 20)
    mean_price = df[df['location'].isin(top20)].groupby('location')['price(L)'].mean()
    mean_price = mean_price.sort_values(ascending=True)
    fig1, ax1 = plt.subplots(figsize=(12, 8))
    mean_price.plot(kind='barh', color='C0', ax=ax1)
    ax1.set_title('Mean Price (L) by Location — Top 20 by Count')
    ax1.set_xlabel('Mean price (Lakhs)')
    plt.tight_layout()
    out1 = os.path.join(script_directory, 'plot_mean_price_per_location.png')
    fig1.savefig(out1)
    plt.close(fig1)

    # 2) Boxplot price by location (top 10)
    top10 = location_counts.head(10).index.tolist()
    fig2, ax2 = plt.subplots(figsize=(14, 8))
    sns.boxplot(data=df[df['location'].isin(top10)], x='price(L)', y='location', ax=ax2, orient='h')
    ax2.set_title('Price Distribution by Location (Top 10 locations by count)')
    ax2.set_xlabel('Price (Lakhs)')
    plt.tight_layout()
    out2 = os.path.join(script_directory, 'boxplot_price_by_location.png')
    fig2.savefig(out2)
    plt.close(fig2)

    # 3) Scatter: area_insqft vs price(L) with trendline
    fig3, ax3 = plt.subplots(figsize=(10, 8))
    sns.scatterplot(data=df, x='area_insqft', y='price(L)', alpha=0.6, s=40, edgecolor=None)
    try:
        sns.regplot(data=df, x='area_insqft', y='price(L)', scatter=False, ax=ax3, color='red', line_kws={'linewidth':1.5})
    except Exception:
        # If regression fails (e.g., due to dtype), skip the line
        pass
    ax3.set_title('Area (sqft) vs Price (Lakhs)')
    ax3.set_xlabel('Area (sqft)')
    ax3.set_ylabel('Price (Lakhs)')
    plt.tight_layout()
    out3 = os.path.join(script_directory, 'scatter_area_vs_price.png')
    fig3.savefig(out3)
    plt.close(fig3)

    # Also report cheapest and most expensive by mean across all locations
    mean_by_loc = df.groupby('location')['price(L)'].mean()
    cheapest = mean_by_loc.idxmin()
    most_expensive = mean_by_loc.idxmax()
    print('\nPlots saved:')
    print(f'  {out1}')
    print(f'  {out2}')
    print(f'  {out3}')
    print(f"Cheapest location by mean price: {cheapest} ({mean_by_loc.min():.2f} L)")
    print(f"Most expensive location by mean price: {most_expensive} ({mean_by_loc.max():.2f} L)")


if __name__ == '__main__':
    main()
import pandas as pd
import os

# Determine the script directory safely and locate the CSV robustly.
# Fix common typos: use __file__ (not _file_) and try multiple candidate names.
script_directory = os.path.dirname(os.path.abspath(__file__))

# List of likely filenames (some in this repo have slightly different spellings/case)
candidate_names = [
    'Hyderbad_House_price.csv',
    'Hyderabad_House_price.csv',
    'hyderabad_House_price.csv',
    'Hyderbad_House_price.csv'
]

file_path = None
for name in candidate_names:
    p = os.path.join(script_directory, name)
    if os.path.exists(p):
        file_path = p
        import argparse
        import pandas as pd
        import os


        def find_csv_file(script_directory: str):
            """Find a likely CSV file in the script directory.

            Returns absolute path or None if not found.
            """
            candidate_names = [
                'Hyderbad_House_price.csv',
                'Hyderabad_House_price.csv',
                'hyderabad_House_price.csv',
                'Hyderbad_House_price.csv'
            ]

            for name in candidate_names:
                p = os.path.join(script_directory, name)
                if os.path.exists(p):
                    return p

            csv_files = [f for f in os.listdir(script_directory) if f.lower().endswith('.csv')]
            matches = [f for f in csv_files if ('house' in f.lower() or 'price' in f.lower())]
            if matches:
                return os.path.join(script_directory, matches[0])
            if csv_files:
                return os.path.join(script_directory, csv_files[0])
            return None


        def load_data(file_path):
            return pd.read_csv(file_path)


        def main():
            parser = argparse.ArgumentParser(description='Load and inspect Hyderabad house price data')
            parser.add_argument('--show-all', action='store_true', help='Print all rows of the dataframe (can be large)')
            parser.add_argument('--save-pickle', action='store_true', help='Save loaded dataframe to hyderabad_data.pkl for faster reloads')
            parser.add_argument('--describe', action='store_true', help='Print df.describe()')
            parser.add_argument('--info', action='store_true', help='Print df.info()')
            args = parser.parse_args()

            script_directory = os.path.dirname(os.path.abspath(__file__))
            file_path = find_csv_file(script_directory)

            try:
                if file_path is None:
                    raise FileNotFoundError('No CSV file found in script directory')

                print(f"Using data file: {file_path}")
                df = load_data(file_path)

                print("🏆 GRAND MASTER STATUS: DATA LOADED!")
                print("-" * 40)
                print(f"Total Houses Found: {len(df)}")
                print(f"Information Categories: {list(df.columns)}")
                print("-" * 40)

                # Basic diagnostics
                print('Memory usage (MB):', round(df.memory_usage(deep=True).sum() / 1024**2, 2))
                missing = df.isna().sum()
                print('Missing values per column:')
                print(missing[missing > 0].to_string() if missing.any() else 'No missing values')

                if args.info:
                    print('\nDataFrame info:')
                    df.info()

                if args.describe:
                    print('\nDataFrame describe():')
                    print(df.describe(include='all'))

                if args.show_all:
                    # Temporarily set pandas option to display all rows
                    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                        print(df)
                else:
                    print('\nFirst 5 rows:')
                    print(df.head(5))

                if args.save_pickle:
                    pkl_path = os.path.join(script_directory, 'hyderabad_data.pkl')
                    df.to_pickle(pkl_path)
                    print(f'Saved dataframe to: {pkl_path}')

            except Exception as e:
                print(f"❌ ERROR: {e}")
                if file_path:
                    print(f"Tried file: {file_path}")
                print("Tips: ensure the CSV is in the same folder as this script and has a name like 'Hyderabad_House_price.csv' or contains 'house'/'price' in its filename.")


        if __name__ == '__main__':
            main()