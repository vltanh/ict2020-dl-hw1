import os

import numpy as np


def generate_data(N):
    X = np.random.uniform(-5, 5, size=(N, 2)).astype(np.float32)
    y = ((X ** 2).sum(1) <= 2.5 ** 2).astype(np.int8)
    return X, y


if __name__ == '__main__':
    out_dir = 'data'
    N1, N2, N3 = 100000, 20000, 20000
    seed = 3698

    np.random.seed(seed)

    print('Generating...', end=' ')
    X_train, y_train = generate_data(N1)
    X_val, y_val = generate_data(N2)
    X_test, y_test = generate_data(N3)
    print('Done!')

    if not os.path.exists(out_dir):
        print('Directory does not exist. Creating...', end=' ')
        os.makedirs(out_dir, exist_ok=True)
        print('Done!')

    print('Saving files...', end=' ')
    fns = ['X_train', 'y_train', 'X_val', 'y_val', 'X_test', 'y_test']
    arrs = [X_train, y_train, X_val, y_val, X_test, y_test]
    for fn, arr in zip(fns, arrs):
        np.save(out_dir + f'/{fn}.npy', arr)
    print('Done!')
