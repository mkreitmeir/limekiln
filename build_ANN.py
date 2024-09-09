import pandas as pd
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pickle
from datetime import datetime
import os

def main(n_neurons):
    # create folder for results
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d-%H-%M-%S")
    results_base_path = f'out/{n_neurons}neurons/{timestamp}/'
    if not os.path.exists(results_base_path):
        os.makedirs(results_base_path)

    ## load and preprocess data
    data = pd.read_csv('out/data.csv', index_col=0).dropna(axis=0, how='any')
    X = data.loc[:, ['dotM_LS', 'dotM_airCombustion', 'dotM_CH4', 'dotM_airCool']]
    y = data.loc[:, ['X', 'T_co']]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    xScaler = StandardScaler()
    X_train_scaled = xScaler.fit_transform(X_train.values)
    X_test_scaled = xScaler.transform(X_test)
    yScaler = StandardScaler()
    y_train_scaled = yScaler.fit_transform(y_train.values)

    ## create and train ANN
    ann = keras.models.Sequential()
    ann.add(keras.layers.Dense(n_neurons, name='hidden', input_shape=(4,), activation='sigmoid'))
    ann.add(keras.layers.Dense(2, name='output'))
    ann.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.MeanSquaredError())
    ann.summary()

    def get_learning_rate(epoch, lr):
        lr_init = 0.1
        lr_factor = 0.1 ** (epoch / 100)
        return lr_init * lr_factor
    lr_scheduler = keras.callbacks.LearningRateScheduler(get_learning_rate)

    csv_logger = keras.callbacks.CSVLogger(os.path.join(results_base_path, 'log.csv'), append=True, separator=',')

    history = ann.fit(X_train_scaled, y_train_scaled, epochs=250, callbacks=[lr_scheduler,csv_logger], 
                verbose=1, batch_size=16, validation_split=0.2)

    ## results
    # some plots
    plt.figure()
    for k, v in history.history.items():
        plt.plot(v, label=k)
    plt.legend()
    plt.xlabel('epoch')
    plt.yscale('log')
    plt.savefig(os.path.join(results_base_path, 'log.png'))

    # parity plots
    y_test_pred = ann.predict(X_test_scaled,verbose=0)
    y_test_pred = yScaler.inverse_transform(y_test_pred)
    y_train_pred = ann.predict(X_train_scaled,verbose=0)
    y_train_pred = yScaler.inverse_transform(y_train_pred)
    mape = keras.metrics.mean_absolute_percentage_error
    mse = keras.metrics.mean_squared_error
    with open(os.path.join(results_base_path, 'metrics.txt'), 'w') as fh:
        for i in range(y.shape[1]):
            mape_train = mape(y_train.iloc[:, i], y_train_pred[:, i])
            mape_test = mape(y_test.iloc[:, i], y_test_pred[:, i])
            fh.write(f'MAPE of {y.columns[i]}. train: {mape_train}%; test: {mape_test}%.\n')
            rmse_train = mse(y_train.iloc[:, i], y_train_pred[:, i]) ** 0.5
            rmse_test = mse(y_test.iloc[:, i], y_test_pred[:, i]) ** 0.5
            fh.write(f'RMSE of {y.columns[i]}. train: {rmse_train}; test: {rmse_test}.\n')

    fig, axs = plt.subplots(1, y.shape[1])
    axs = axs.ravel()
    for i, ax in enumerate(axs):
        match i:
            case 0:
                ax.set_title('Conversion')
            case 1:
                ax.set_title('Channel temperature')
        y_mean = y_test.iloc[:, i].mean()
        ax.axline(xy1=(y_mean,y_mean), slope=1, c='k')
        ax.axline(xy1=(y_mean,1.02*y_mean), slope=1, c='k', ls=':')
        ax.axline(xy1=(y_mean,0.98*y_mean), slope=1, c='k', ls=':')
        ax.scatter(y_test.iloc[:, i], y_test_pred[:, i])
        ax.set_xlabel('True value')
        ax.set_ylabel('Predicted value')
        ax.axis('square')
    fig.tight_layout()
    fig.savefig(os.path.join(results_base_path, 'parity.png'))

    # save
    out = pd.DataFrame({'X_true': y_test.iloc[:, 0],
                        'X_pred': y_test_pred[:, 0],
                        'T_true': y_test.iloc[:, 1],
                        'T_pred': y_test_pred[:, 1]})
    out.to_csv(os.path.join(results_base_path, 'parity.csv'), index=False)

    ann_dict = {'ann': ann, 'xScaler': xScaler, 'yScaler': yScaler, 'X_train': X_train, 'y_train': y_train, 'X_test': X_test, 'y_test': y_test}
    with open(os.path.join(results_base_path, 'ann.pkl'), 'wb') as fh:
        pickle.dump(ann_dict, fh)
    # plt.show()


if __name__ == '__main__':
    for n_neurons in range(10, 11):
        for i in range(3):
            main(n_neurons)