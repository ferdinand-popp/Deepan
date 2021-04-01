import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter


def create_survival_plot(df=None, path_csv=None):
    """
    Survival analysis on a dataframe containing patients ID, vital_status, days_to_death and grouped label from pipeline
    Returns plot figure
    Based on https://pub.towardsai.net/survival-analysis-with-python-tutorial-how-what-when-and-why-19a5cfb3c312
    and the lifelines package.
    """
    if df is None:
        # Read the dataset:
        df = pd.read_csv(path_csv, sep="\t", index_col=0)

    # kick -1 label unsorted, and empty death or survival
    df = df[df.labels != -1]
    df = df.dropna(how='any', axis=0)

    # Organize our data:
    df.loc[df.vital_status == 0, 'dead'] = 0
    df.loc[df.vital_status == 1, 'dead'] = 1

    # Create two objects for groups:
    groups = set(df.labels)

    fitters = []
    # Dividing data into groups:
    for k in groups:
        class_member_mask = (df.labels == k)
        df_groups = df[class_member_mask]
        fitters.append(
            KaplanMeierFitter().fit(durations=df["days_to_death"], event_observed=df_groups["dead"], label=k))

    print(fitters[0].event_table)  # event_table, predict(days), survival_function_, cumulative_density_

    fig = plt.figure()
    for fit in fitters:
        fit.plot()
    plt.xlabel("Days Passed")
    plt.ylabel("Survival Probability")
    plt.title("KMF")

    return fig
