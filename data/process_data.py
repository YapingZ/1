# import libraries

import pandas as pd
import sys

from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    # load csv file
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    # merge datasets
    df = pd.merge(messages, categories, on='id')
    return df


def clean_data(df):

    # create a dataframe of the 36 individual category columns
    categories1 = df.categories.str.split(';', expand=True)

    # use this row to extract a list of new column names for categories.
    row = categories1.iloc[0]
    category_colnames = row.apply(lambda x: x[:-2])
    print(category_colnames)

    # rename the columns of `categories`
    categories1.columns = category_colnames
    for column in categories1:
        # set each value to be the last character of the string
        categories1[column] = categories1[column].str[-1]
        # convert column from string to numeric
        categories1[column] = categories1[column].astype(int)

    # drop the original categories column from `df`
    df.drop('categories', axis=1, inplace=True)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories1], axis=1)
    # df.duplicated().sum()
    df.drop_duplicates(inplace=True)

    return df


def save_data(df, database_name):

    engine = create_engine('sqlite:///{}'.format(database_name))
    df.to_sql('DisasterMessage', engine, index=False)


def main():
    if len(sys.argv) == 4:
        messages_filepath, categories_filepath, database_name = sys.argv[1:]
        print('Loading data... \n Messages: {} \n  Categories: {} \n '
              .format(messages_filepath, categories_filepath))

        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data... \n ')
        df = clean_data(df)

        print('Saving data to database...\n Database: {}'.format(database_name))
        save_data(df, database_name)

        print('Congratulations! Your data is cleaned and saved to database!')

    else:
        print('Please type the right filepathes!')


if __name__ == '__main__':
    main()
