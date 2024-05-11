import streamlit as st
import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import plotly.express as px

# Load dataset 
df = pd.read_csv("games.csv")

# Defining rating categories
rating_categories = [
    "< 1000", "1000-1100", "1100-1200", "1200-1300",
    "1300-1400", "1400-1500", "1500-1600", "1600-1700",
    "1700-1800", "1800-1900", "1900-2000", "2000-2100",
    "2100-2200", "2200-2300", "2300-2400", "> 2400"
]

def clean_and_process_df(df):
    
    
    selected_columns = ['id', 'rated', 'winner', 'increment_code', 'white_rating', 'black_rating', 'moves', 'opening_name']
    df = df[selected_columns].copy()  # Use .copy() to avoid SettingWithCopyWarning

    # Simplify replacing values using map for better readability
    df['result'] = df['winner'].map({"white": "White won", "black": "Black won", "draw": "Draw"})

    # Convert True/False to 1/0 
    df['rated'] = df['rated'].astype(int)

    # Map 'winner' to numeric values
    df['winner_dbl'] = df['winner'].map({'white': 1, 'black': 0, 'draw': 0.5})

    # Calculate rating difference between white and black
    df['rating_diff'] = df['white_rating'] - df['black_rating']

    # Extracting base time from the increment code, handling NaN values appropriately
    df['base_time'] = df['increment_code'].str.extract(r'^(\d+)')
    df['base_time'] = df['base_time'].fillna(0).astype(int)  # Fill NaN with 0 then convert to int

    # Filter out rows where base time is 0
    df = df[df['base_time'] != 0]

    # Determine if the game had increment
    df['has_increment'] = (~df['increment_code'].str.contains(r'\+0')).astype(int)  # 0 if '+0' is in increment code, otherwise 1.

    # Extract the first move from the moves column
    df['first_move'] = df['moves'].str.extract(r'^(\S+)')

    # Calculating average rating
    df['avg_rating'] = (df['white_rating'] + df['black_rating']) / 2
    # Creating average rating category with bins e.g. (-inf, 1000], (1000, 2000]
    df['avg_rating_category'] = pd.cut(
        df['avg_rating'], 
        bins=[-np.inf, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, np.inf],
        labels=rating_categories
    )

    # Reset the index of the DataFrame after filtering and modifying
    df.reset_index(drop=True, inplace=True)

    return df

def plot_bar_chart(df, x_column, y_column, color_column, labels, title, color_map, orientation, barmode):
    """
    Returns a customised plotly bar plot.
    """
    fig = px.bar(
        data_frame=df, 
        x=x_column, 
        y=y_column, 
        color=color_column, 
        labels=labels, 
        title=title, 
        template="plotly_white", 
        color_discrete_map=color_map, 
        orientation=orientation,
        barmode = barmode
    )
    fig.update_yaxes(showline=False,showgrid=False)
    fig.update_xaxes(showline=False,showgrid=False)
    return fig

def plot_game_results(df):
    """
    Game results in percentage
    """
    game_result_df = df['result'].value_counts(normalize=True).reset_index()
    game_result_df.columns = ['result', '%']
    game_result_df['%'] = round(100 * game_result_df['%'], 2)

    # Creating and showing the bar plot
    fig = plot_bar_chart(
        df=game_result_df, 
        x_column='result', 
        y_column='%', 
        color_column=None, 
        labels={"result": "Game result", "%": "Proportion of wins (%)"},
        title="Chess Game Results", 
        color_map=None, 
        orientation=None, 
        barmode=None
    )
    return fig

def plot_by_rating(df):
    """
    Game results by average rating of players
    """
    # Grouping and pivoting the data
    rating_result_pivot = df.pivot_table(values='id', index='avg_rating_category', columns='result', aggfunc='count')

    # Normalizing the data by row to get percentages
    rating_result_percentage = rating_result_pivot.div(rating_result_pivot.sum(axis=1), axis=0) * 100

    # Resetting the index for Plotly
    rating_result_df = rating_result_percentage.reset_index()

    # Melting the DataFrame for easier plotting
    rating_df_melt = rating_result_df.melt(
        id_vars=['avg_rating_category'], 
        value_vars=['White won', 'Draw', 'Black won'],
        var_name='result',
        value_name='%'
    )

    # Creating and showing the bar plot
    fig = plot_bar_chart(
        df=rating_df_melt, 
        x_column="%", 
        y_column="avg_rating_category", 
        color_column='result', 
        labels={
            "avg_rating_category": "Player Rating", 
            "%": "Proportion of wins (%)", 
            "result": "Game result"
        }, 
        title="Chess Game Results vs Player Rating", 
        color_map={
            "White won": "whitesmoke", 
            "Draw": "lightgrey", 
            "Black won": "dimgrey"
        }, 
        orientation='h', 
        barmode='relative'
    )
    return fig

def plot_by_time(df):
    """
    Game results by game's base time
    """
    # Grouping and creating a DataFrame for game results by base time
    base_time_results = df.groupby(['base_time', 'result']).size().reset_index(name='count')

    # Filtering out games with base time more than 20 minutes
    filtered_results = base_time_results[base_time_results.base_time <= 20]

    pivot_results = pd.pivot_table(filtered_results, values='count', index='base_time', columns='result')
    plot_df = pivot_results.reset_index().fillna(0)

    melted_df = pd.melt(
        plot_df,
        id_vars=['base_time'], 
        value_vars=['White won', 'Draw', 'Black won'],
        var_name='result',
        value_name='count'
    )
    melted_df['Percentage'] = round(100 * melted_df['count'] / melted_df.groupby('base_time')['count'].transform('sum'), 2)

    # Create chart
    fig = plot_bar_chart(
        df=melted_df, 
        x_column="base_time", 
        y_column="Percentage", 
        color_column="result", 
        labels={
            "base_time": "Base time (min)", 
            "Percentage": "Proportion of wins (%)", 
            "result": "Game result"
        }, 
        title="Chess Game Results vs Base Time of Game", 
        color_map={"White won": "whitesmoke", "Draw": "lightgrey", "Black won": "dimgrey"},
        orientation='v', 
        barmode="relative"
    )
    return fig

def get_selected_ratings(min_rating, max_rating):
    """
    Get list of selected ratings by the selected rating range.
    """
    min_rating_index = rating_categories.index(min_rating)
    max_rating_index = rating_categories.index(max_rating) 

    selected_ratings = rating_categories[min_rating_index:max_rating_index + 1] # Include of the selected max rating

    return selected_ratings

def plot_first_move_count(df, num, min_rating, max_rating):
    """
    Returns a plotly histogram of the top {num} first moves played in
    the selected range of rated games.
    """
    selected_ratings = get_selected_ratings(min_rating, max_rating)
    filtered_df = df[df['avg_rating_category'].isin(selected_ratings)]
    num_games = len(filtered_df.index)

    # Count by first move
    first_move_df = filtered_df.groupby('first_move').size().reset_index(name='count')
    first_move_df = first_move_df[first_move_df['count'] > 0]

    # Sort the DataFrame by 'count' column in descending order and reset row index
    first_move_df = first_move_df.sort_values(by='count', ascending=False)
    first_move_df = first_move_df.reset_index(drop=True)

    # Get the top # openings played
    first_move_df = first_move_df.head(num)

    fig = plot_bar_chart(
        df=first_move_df, 
        x_column='first_move',
        y_column='count',
        color_column=None, 
        labels={"first_move": "First move", "count": "Number of times played"},
        title=f"Top {num} First Moves Played in {min_rating} to {max_rating} Rated Games \
            <br><sup>(from {num_games} Lichess Games)</sup>", 
        color_map=None, orientation=None, barmode=None
    )
    return fig

def plot_by_first_move(df, num, min_rating, max_rating):
    selected_ratings = get_selected_ratings(min_rating, max_rating)
    filtered_df = df[df['avg_rating_category'].isin(selected_ratings)]
    num_games = len(filtered_df.index)

    # Grouping by first move and result, then counting occurrences
    result_df = filtered_df.groupby(['first_move', 'result']).size().reset_index(name='count')
    result_df = result_df[result_df['count'] > 0]

    result_pivot = pd.pivot_table(
        result_df, 
        values='count', 
        index='first_move', 
        columns='result', 
        fill_value=0
    )
    # Add a column for total count of games per first move
    result_pivot['Total Count'] = result_pivot.sum(axis=1)
    # Sort by descending count of total games
    result_pivot.sort_values(by='Total Count', ascending=False, inplace=True)
    result_pivot.reset_index(inplace=True)

    melted_df = pd.melt(
        result_pivot,
        id_vars=['first_move'], 
        value_vars=['White won', 'Draw', 'Black won'],
        var_name='result',
        value_name='count'
    )
    melted_df['percentage'] = round(100 * melted_df['count'] / melted_df.groupby('first_move')['count'].transform('sum'), 2)
    # Sorting by 'result' as 'White won' and then by 'Percentage' in descending order
    melted_df.sort_values(by=['result', 'count'], ascending=[False, True], inplace=True)

    # Creating and showing the bar plot for game results vs first move
    fig = plot_bar_chart(
        df=melted_df, 
        x_column="percentage",
        y_column="first_move",
        color_column='result', 
        labels={"first_move": "First Move", "Percentage": "Proportion of Wins (%)", "result": "Game Result"}, 
        title="Game Results vs First Move of the Game", 
        color_map={"White won": "whitesmoke", "Draw": "lightgrey", "Black won": "dimgrey"}, 
        orientation='h', barmode="relative"
    )
    return fig

def plot_top_openings(df, num, min_rating, max_rating):
    openings_df = df.groupby('opening_name').size().reset_index(name='count')

    # Sort the DataFrame by 'count' column in descending order and reset row index
    openings_df = openings_df.sort_values(by='count', ascending=False).reset_index(drop=True)

    # Get the top 10
    openings_df = openings_df.head(10)

    fig = plot_bar_chart(
        df=openings_df, 
        x_column='opening_name', 
        y_column='count', 
        color_column=None, 
        labels={"opening_name": "Opening", "count": "Number of times played"},
        title="Top {num} Openings Played", 
        color_map=None, orientation=None, barmode=None
    )
    return fig

def get_player_ratings(df):
    return

df = clean_and_process_df(df)

# game_result_chart = plot_game_results(df)
# st.plotly_chart(game_result_chart)

# by_rating_chart = plot_by_rating(df)
# st.plotly_chart(by_rating_chart)

# by_time_chart = plot_by_time(df)
# st.plotly_chart(by_time_chart)

# Get min/max rating categories to filter by
min_rating, max_rating = st.select_slider(
    "Select a range of average rating of games",
    options=rating_categories,
    value=(rating_categories[0], rating_categories[-1]))

MIN_NUM_RESULTS, MAX_NUM_RESULTS = 1, 30
num_results = st.slider("Number of results to display", MIN_NUM_RESULTS, MAX_NUM_RESULTS, 10)

first_move_histogram = plot_first_move_count(df, num_results, min_rating, max_rating)
st.plotly_chart(first_move_histogram)

by_first_move_chart = plot_by_first_move(df, num_results, min_rating, max_rating)
st.plotly_chart(by_first_move_chart)