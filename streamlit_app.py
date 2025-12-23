import os
from typing import Optional

import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


def find_cleaned_pickle(script_directory: str) -> Optional[str]:
    p1 = os.path.join(script_directory, 'hyderabad_data_cleaned.pkl')
    if os.path.exists(p1):
        return p1
    # fallback
    csv_fallback = os.path.join(script_directory, 'hyderabad_data_cleaned.csv')
    if os.path.exists(csv_fallback):
        return csv_fallback
    return None


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    if path.endswith('.pkl'):
        return pd.read_pickle(path)
    return pd.read_csv(path)


def plot_mean_price_by_location(df: pd.DataFrame, top_n: int = 20):
    counts = df['location'].value_counts()
    top = counts.head(top_n).index.tolist()
    mean_price = df[df['location'].isin(top)].groupby('location')['price(L)'].mean().sort_values()
    fig, ax = plt.subplots(figsize=(10, max(4, top_n * 0.3)))
    mean_price.plot(kind='barh', ax=ax, color='C0')
    ax.set_xlabel('Mean price (Lakhs)')
    ax.set_title(f'Mean price by location (top {top_n} locations by count)')
    plt.tight_layout()
    return fig


def plot_boxplot_by_location(df: pd.DataFrame, top_n: int = 10):
    counts = df['location'].value_counts()
    top = counts.head(top_n).index.tolist()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df[df['location'].isin(top)], x='price(L)', y='location', ax=ax, orient='h')
    ax.set_xlabel('Price (Lakhs)')
    ax.set_title(f'Price distribution by location (top {top_n})')
    plt.tight_layout()
    return fig


def plot_area_vs_price(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=df, x='area_insqft', y='price(L)', alpha=0.6, s=40, edgecolor=None)
    try:
        sns.regplot(data=df, x='area_insqft', y='price(L)', scatter=False, ax=ax, color='red', line_kws={'linewidth':1.2})
    except Exception:
        pass
    ax.set_xlabel('Area (sqft)')
    ax.set_ylabel('Price (Lakhs)')
    ax.set_title('Area vs Price')
    plt.tight_layout()
    return fig


def main():
    st.set_page_config(page_title='Hyderabad Houses — Demo', layout='wide')
    st.title('Hyderabad House Prices — Demo')

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = find_cleaned_pickle(script_dir)
    if data_path is None:
        st.error('Cleaned dataset not found. Run the cleaning script first to create `hyderabad_data_cleaned.pkl`.')
        return

    df = load_data(data_path)

    st.sidebar.header('Controls')
    top_n = st.sidebar.slider('Top N locations by count (mean bar chart)', min_value=5, max_value=50, value=20)
    top_box = st.sidebar.slider('Top N locations for boxplot', min_value=3, max_value=20, value=10)
    show_table = st.sidebar.checkbox('Show data table', value=False)
    download_zip = st.sidebar.checkbox('Show download link for plots zip', value=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.header('Mean Price by Location')
        fig1 = plot_mean_price_by_location(df, top_n=top_n)
        st.pyplot(fig1)

        st.header('Area vs Price')
        fig3 = plot_area_vs_price(df)
        st.pyplot(fig3)

    with col2:
        st.header('Boxplot — Top Locations')
        fig2 = plot_boxplot_by_location(df, top_n=top_box)
        st.pyplot(fig2)

        st.markdown('---')
        st.write('Basic stats:')
        st.write(df[['price(L)']].describe())

        if download_zip:
            zip_path = os.path.join(script_dir, 'hyd_plots.zip')
            if os.path.exists(zip_path):
                with open(zip_path, 'rb') as f:
                    st.download_button('Download all plots (zip)', data=f, file_name='hyd_plots.zip')
            else:
                st.info('Zip file not found; the script will generate plots on the fly.')

    if show_table:
        st.header('Data (preview)')
        st.dataframe(df)


if __name__ == '__main__':
    main()
