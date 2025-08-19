from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import numpy as np
import os
import pandas as pd

# 激活环境，运行： python -m scripts.generate_training_data --output_dir=data/METR-LA --traffic_df_filename=data/metr-la.h5 --seq_len=12 --horizon=3

def generate_graph_seq2seq_io_data(
        df, x_offsets, y_offsets, add_time_in_day=True, add_day_in_week=False, scaler=None
):
    """
    Generate samples from
    :param df:
    :param x_offsets:
    :param y_offsets:
    :param add_time_in_day:
    :param add_day_in_week:
    :param scaler:
    :return:
    # x: (epoch_size, input_length, num_nodes, input_dim)
    # y: (epoch_size, output_length, num_nodes, output_dim)
    """

    num_samples, num_nodes = df.shape
    data = np.expand_dims(df.values, axis=-1)
    data_list = [data]
    if add_time_in_day:
        time_ind = (df.index.values - df.index.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
        time_in_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
        data_list.append(time_in_day)
    if add_day_in_week:
        day_in_week = np.zeros(shape=(num_samples, num_nodes, 7))
        day_in_week[np.arange(num_samples), :, df.index.dayofweek] = 1
        data_list.append(day_in_week)

    data = np.concatenate(data_list, axis=-1)
    # epoch_len = num_samples + min(x_offsets) - max(y_offsets)
    x, y = [], []
    # t is the index of the last observation.
    min_t = abs(min(x_offsets))
    max_t = abs(num_samples - abs(max(y_offsets)))  # Exclusive
    for t in range(min_t, max_t):
        x_t = data[t + x_offsets, ...]
        y_t = data[t + y_offsets, ...]
        x.append(x_t)
        y.append(y_t)
    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)
    return x, y

def get_0_1_array(array,rate=0.2):
    zeros_num = int(array.size * rate)
    new_array = np.ones(array.size)
    new_array[:zeros_num] = 0
    np.random.shuffle(new_array)
    re_array = new_array.reshape(array.shape)
    return re_array

def generate_train_val_test(args):
    df = pd.read_hdf(args.traffic_df_filename)
    if args.mask_rate > 0.0 and args.mask_rate <= 1.0:
        print(f"Original df shape: {df.shape}, Non-zero count: {np.count_nonzero(df.values)}")
        print(f"Applying mask with rate {args.mask_rate} to the dataframe values.")
        
        data_values = df.values # Get underlying numpy array
        mask = get_0_1_array(data_values, args.mask_rate)
        
        # Apply mask: where mask is 0, data becomes 0.
        # This assumes missing data is represented as 0.
        # Note: If df.values was int, data_values * mask would result in float.
        # If df.values was float, result remains float.
        masked_values = data_values * mask
        
        # Update the DataFrame with masked values.
        # This preserves the DataFrame structure (index, columns).
        df = pd.DataFrame(masked_values, index=df.index, columns=df.columns)
        print(f"Masked df shape: {df.shape}, Non-zero count after masking: {np.count_nonzero(df.values)}")
        print(f"Number of zeros in mask: {mask.size - np.count_nonzero(mask)} (expected: {int(mask.size * args.mask_rate)})")
    # 0 is the latest observed sample.
    #x_offsets = np.sort(
        # np.concatenate(([-week_size + 1, -day_size + 1], np.arange(-11, 1, 1)))
    #    np.concatenate((np.arange(-11, 1, 1),)))
    # Predict the next one hour
    #y_offsets = np.sort(np.arange(1, 13, 1))
    x_offsets = np.sort(np.arange(-args.seq_len + 1, 1, 1))
    y_offsets = np.sort(np.arange(1, args.horizon+1, 1))
    # x: (num_samples, input_length, num_nodes, input_dim)
    # y: (num_samples, output_length, num_nodes, output_dim)
    x, y = generate_graph_seq2seq_io_data(
        df,
        x_offsets=x_offsets,
        y_offsets=y_offsets,
        add_time_in_day=True,
        add_day_in_week=False,
    )

    print("x shape: ", x.shape, ", y shape: ", y.shape)
    # Write the data into npz file.
    # num_test = 6831, using the last 6831 examples as testing.
    # for the rest: 7/8 is used for training, and 1/8 is used for validation.
    num_samples = x.shape[0]
    num_test = round(num_samples * 0.2)
    num_train = round(num_samples * 0.7)
    num_val = num_samples - num_test - num_train

    # train
    x_train, y_train = x[:num_train], y[:num_train]
    # val
    x_val, y_val = (
        x[num_train: num_train + num_val],
        y[num_train: num_train + num_val],
    )
    # test
    x_test, y_test = x[-num_test:], y[-num_test:]

    for cat in ["train", "val", "test"]:
        _x, _y = locals()["x_" + cat], locals()["y_" + cat]
        print(cat, "x: ", _x.shape, "y:", _y.shape)

        filename = f"{cat}_seq_len_{args.seq_len}_horizon_{args.horizon}_{args.mask_rate}.npz"
        np.savez_compressed(
            os.path.join(args.output_dir, filename),
            x=_x,
            y=_y,
            x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]),
            y_offsets=y_offsets.reshape(list(y_offsets.shape) + [1]),
        )


def main(args):
    print("Generating training data")
    generate_train_val_test(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir", type=str, default="data/PEMS-BAY", help="Output directory."
    )
    parser.add_argument(
        "--traffic_df_filename",
        type=str,
        default="data/pems-bay.h5",
        help="Raw traffic readings.",
    )
    parser.add_argument(
    "--seq_len",
    type=int,
    default=12,
    help="Length of the input sequence."
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=12,
        help="Length of the output sequence."
    )
    parser.add_argument(
        "--mask_rate",
        type=float,
        default=0.2,
        help="Length of the output sequence."
    )
    args = parser.parse_args()
    main(args)
