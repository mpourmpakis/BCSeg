import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import Pipeline
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.metrics import mean_absolute_error, make_scorer


def load_dataset():
    """
    Loads complete Eseg dataset as dataframe

    Returns:
    df (pd.DataFrame): dataframe of training and target data
                       NOTE: see /data/README.md
                             for more info on data
    """
    # get path to complete dataset
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'data',
                        'dataset.csv')

    # pandas DataFrame object
    df = pd.read_csv(path)

    return df


def get_training_data(df=None):
    """
    Get the training (5 features) and target data to train Eseg model

    KArgs:
    df (pd.DataFrame): if not given, call load_data() to get dataset
                       (Default: None)

    Returns:
    X, y (np.ndarray, np.ndarray): training and target data for Eseg model
    """
    if df is None:
        df = load_dataset()

    # 5 input features (X) and target (y, Eseg)
    X = df[['diff_CEb/cn',
            'gordy_eneg_host',
            'diff_EA',
            'r_dopant',
            'IP_dopant']].values
    y = df['Eseg'].values

    return X, y


def get_eseg_model():
    """
    Creates trained Eseg model pipeline, including:
    - StandardScaler applied to inputs
    - 2nd-order polynomial Kernel Ridge Regression

    Returns:
    model (Pipeline): Trained Eseg model
                     - make predictions using model.predict(x)
    """
    # define training data (X) and targets (y = Eseg)
    X, y = get_training_data()

    # create standard scaler
    scaler = StandardScaler()

    # create kernel ridge estimator (with tuned hyperparameters)
    kernel = KernelRidge(kernel='poly', degree=2, alpha=0.002, gamma=0.02)

    # create model pipeline
    model = Pipeline(steps=[('StandardScaler', scaler),
                            ('KernelRidge', kernel)])

    # train and return model
    model.fit(X, y)

    return model


def reproduce_parity_plot():
    """
    Reproduces parity plot of Eseg model predictions
    from our paper. LOOCV-MAE = 0.220 eV

    Returns:
    fig, ax (Figure, Axes): matplotlib figure and axes objects for
                            additional modification and/or saving
    """
    df = load_dataset()
    X, y = get_training_data(df)
    eseg_model = get_eseg_model()

    # calcualte LOOCV-MAE score
    loocv_mae_krr = cross_val_score(eseg_model, X, y,
                                    scoring=make_scorer(mean_absolute_error),
                                    cv=LeaveOneOut())

    # calculate predicted Eseg
    df['predicted_Eseg'] = eseg_model.predict(X)

    # create parity plot
    fig, ax = plt.subplots()

    # remove spines on top and right of plot
    ax.axes.spines['top'].set_visible(False)
    ax.axes.spines['right'].set_visible(False)

    # plot data and color by Host metal
    for host, group in df.groupby('Host'):
        ax.scatter(group.Eseg, group.predicted_Eseg, label=f'{host} Host',
                   alpha=0.8)

    # add parity line
    parity = np.linspace(y.min(), y.max())
    ax.plot(parity, parity, color='k', zorder=-50)

    # add axis labels and legend
    ax.set_ylabel('$\\rm E_{seg,model}$ (eV)', fontsize=14)
    ax.set_xlabel('$\\rm E_{seg,DFT}$ (eV)', fontsize=14)

    ax.legend(
        title=f'2nd order poly KRR\nLOOCV-MAE= {loocv_mae_krr.mean():0.3f} eV',
        ncol=2,
        frameon=False)

    # apply tight_layout
    fig.tight_layout()

    return fig, ax


if __name__ == '__main__':
    fig, ax = reproduce_parity_plot()
    plt.show()
