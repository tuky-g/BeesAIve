# USAGE
# python bs_label_generator.py

# Import the necessary packages
import os
import argparse
from glob import glob
import pandas as pd
from sklearn.model_selection import train_test_split

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", type=str,
                default="../dataset", help="path to the dataset")
ap.add_argument("-c", "--csv", type=str,
                default="../bs_labels.csv", help="path to save csv")
args = vars(ap.parse_args())

# Import images path
image_files = sorted(glob(os.path.join(args["dataset"], '**', '*.*'),
                          recursive=True))

# Count total images found
print(f"Dataset contains {len(image_files)} images")

# Create DataFrame with labels
df = pd.DataFrame(image_files, columns=['path'])
df['file_name'] = df['path'].apply(lambda x: os.path.basename(x))
df['label'] = df['path'].apply(lambda x: os.path.basename(os.path.dirname(x)))
df['file_path'] = df['label'] + '/' + df['file_name']
df.drop('path', axis=1, inplace=True)

print("\nLabels found: ", end='')
print(df.label.unique())

print("\nClasses count:")
print(df.label.value_counts())

# Split the data into training and test sets
train, test = train_test_split(df,
                               test_size=0.2,
                               random_state=42,
                               stratify=df['label'])

val, test = train_test_split(test,
                             test_size=0.5,
                             random_state=42,
                             stratify=test['label'])

df_f = pd.concat([train, val, test],
                 keys=['train', 'validation', 'test']).reset_index()
df_f.drop('level_1', axis=1, inplace=True)
df_f = df_f.rename({'level_0': 'subset'}, axis=1)
df_f = df_f[['file_path', 'file_name', 'label', 'subset']]

print("\nSubsets count:")
print(df_f.value_counts('subset'))

# Export CSV
df_f.to_csv(args["csv"], index=False)
