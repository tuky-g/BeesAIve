# USAGE
# python bs_label_generator.py --csv bs_label.csv

# import the necessary packages
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import cv2


def create_piechart(subset='train'):
    df_s = label_df[label_df['subset'] == subset]
    labels = df_s['label'].value_counts().index
    sizes = df_s['label'].value_counts().values

    plt.figure(figsize=[5, 5])
    plt.pie(sizes,
            colors=plt.cm.tab20_r.colors,
            labels=labels,
            autopct='%1.1f%%',
            pctdistance=0.80,
            explode=(0.02, 0.02, 0.02, 0.02),
            textprops={'fontsize': 12})

    centre_circle = plt.Circle((0, 0), 0.60, fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)

    title = f'{subset.capitalize()} subset class distribution'
    plt.title(title, fontsize=16)

    plt.tight_layout()
    plt.show()

    print(''.join(['\n', title]))
    print(df_s['label'].value_counts())


# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--csv", type=str,
                required=True, help="path to labels csv")
args = vars(ap.parse_args())

label_df = pd.read_csv(args["csv"])

# Check null values
print('Null values: ')
print(label_df.isnull().sum())

# Check distribution of training and test sets
print('\nSubsets distribution: ')
print(label_df.value_counts('subset'))

train_perc = round(label_df.value_counts('subset')[
                   'train'] / len(label_df) * 100, 2)
val_perc = round(label_df.value_counts('subset')[
                 'validation'] / len(label_df) * 100, 2)
test_perc = round(label_df.value_counts('subset')[
                  'test'] / len(label_df) * 100, 2)

print(f'\nTrain subset - {train_perc}%')
print(f'Validation subset - {val_perc}%')
print(f'Test subset - {test_perc}%')

# Show distributions
create_piechart()

create_piechart('validation')

create_piechart('test')
