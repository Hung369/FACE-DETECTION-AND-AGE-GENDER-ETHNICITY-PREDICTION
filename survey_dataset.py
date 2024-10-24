import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

def print_head(data):
    print(data.head())

def age_distribution(age_dist):
    plt.figure(figsize=(12, 6))
    plt.bar(age_dist.index, age_dist.values, color='skyblue')
    plt.xlabel('Age')
    plt.ylabel('Count')
    plt.title('Age Distribution')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.show()


def gender_distribution(gender_dist):
    plt.figure(figsize=(6, 5))
    gender_dist.plot(kind='bar', color='lightgreen')
    plt.xlabel('Gender')
    plt.ylabel('Count')
    plt.title('Gender Distribution')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.xticks(rotation=0)
    plt.show()

def ethnicity_distribution(ethnicity_dist):
    plt.figure(figsize=(8, 5))
    ethnicity_dist.plot(kind='bar', color='lightcoral')
    plt.xlabel('Ethnicity')
    plt.ylabel('Count')
    plt.title('Ethnicity Distribution')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.show()


if __name__ == "__main__":
    data = pd.read_csv('./data/age_gender.csv')
    print_head(data)

    print('Total rows: {}'.format(len(data)))
    print('Total columns: {}'.format(len(data.columns)))

    age_dist = data['age'].value_counts().sort_index()
    ethnicity_dist = data['ethnicity'].value_counts().rename(index={0:"White", 1:"Black", 2:"Asian", 3:"Indian", 4:"Others"})
    gender_dist = data['gender'].value_counts().rename(index={0:'Male',1:'Female'})

    age_distribution(age_dist)
    ethnicity_distribution(ethnicity_dist)
    gender_distribution(gender_dist)