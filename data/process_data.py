import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Load two datasets and merge it into single DataFrame

    Args:
        messages_filepath (str): path to messages csv
        categories_filepath (str): path to categories csv

    Return:
        df (DataFrame): merged dataframe
    """

    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
  
    # merge the messages and categories datasets using the common id
    # assign this combined dataset to `df`
    df = messages.merge(categories, how="left" , on="id")

    return df

def clean_data(df):
    """
    Clean DataFrame: Split the values in the `categories` column on the `;` character so that each value becomes a separate column. 
    Rename columns of `categories` with new column names.

    Args:
        df (DataFrame): input DataFrame

    Return:
        df (DataFrame): cleaned DataFrame
    
    """
    
    # create a dataframe of the 36 individual category columns
    categories = pd.Series(df["categories"]).str.split(";", expand=True)

    # select the first row of the categories dataframe
    row = categories.iloc[0]
    
    #extract a list of new column names for categories.
    category_colnames = [x.split("-")[0] for x in row]

    # rename the columns of `categories`
    categories.columns = category_colnames

    #convert category values to just numbers 0 or 1.

    for column in categories:
        # set each value to be the last character of the string if the value not equal 0 or 1 set it to np.nan
        categories[column] = categories[column].apply(lambda x: x.split("-")[1] if int(x.split('-')[1]) < 2 else np.nan)
        
        #check number of misclassified entries
        nans = categories[column].isnull().sum()
        if nans != 0:
            print("Number of misclassified rows: ", nans)
            #impute np.nan with most frequent value in column
            categories[column].fillna(categories[column].value_counts().index[0], inplace=True)
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
   
    #replace `categories` column in `df` with new category columns.
    #drop the categories column from the df dataframe since it is no longer needed
    df.drop("categories", axis=1, inplace=True)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories], axis=1)
    
    # check number of duplicates
    n_duplicates = df.duplicated().sum() 
    print("Initial number of duplicates: ", n_duplicates)
    if n_duplicates !=0:
        # drop duplicates
        df.drop_duplicates(inplace=True)
        print("After duplicates dropping the number of duplicates: ", df.duplicated().sum())

    return df


def save_data(df, database_filename):
    """
    Save clean data set to sqlite database

    Args:
        df (DataFrame): clean DataFrame

    Return:
        None
    """

    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('merged_data', engine, if_exists="replace", index=False, chunksize=75)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
