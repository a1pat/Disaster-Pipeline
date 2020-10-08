import sys
import numpy as py
import pandas as pd
from sqlalchemy import create_engine



def load_data(messages_filepath, categories_filepath):
    '''
    Load the messages and categories data sets
    Inputs:
        messages_filepath: (string) file path to csv file containing messages data
        categories_filepath: (string) file path to csv file containing categories data
    Returns:
        df: pandas dataframe containing messages and categories data
    '''
    # load messages data set
    messages = pd.read_csv(messages_filepath)
    
    # load categories data set
    categories = pd.read_csv(categories_filepath)
    
    # merge data sets
    df = messages.merge(categories, how='inner', on='id')
    
    return df



def clean_data(df):
    '''
    Clean the data set
    Input:
        df: (pandas dataframe) messages and categories data
    Returns:
        df: (pandas dataframe) clean data set
    '''
    
    # create a dataframe of the individual category columns
    categories = df['categories'].str.split(';', expand=True)
    
    # select the first row of the categories dataframe
    row = categories.iloc[0]
    
    # use this to extract a list of new columns for the categories
    category_colnames = []
    for colname in row.str.split('-'):
        category_colnames.append(colname[0])
        
    # rename the columns of the categories dataframe
    categories.columns = category_colnames
    
    # extract just the number for each category (usually 0 or 1)
    for column in categories:
        # set each value to the last character in the string
        clean = []
        for e in categories[column]:
            clean.append(e[-1])
        categories[column] = clean
        categories[column] = pd.to_numeric(categories[column])
        
    # drop the original categories column from df
    df.drop('categories', axis=1, inplace=True)
    
    # concatenate the original categories dataframe with the new categories dataframe
    df = pd.concat([df, categories], axis=1)
    
    # how many duplicate rows are in the data set?
    num_duplicates = df.shape[0] - df.pivot_table(index='id', aggfunc='size').count()
    # drop duplicate rows, if there are any
    if num_duplicates > 0:
        df.drop_duplicates(subset='id', inplace=True)
        
    # drop observations that have a 2 in the 'related' column
    df.drop(df[df['related'] == 2].index, inplace=True)
    
    return df



def save_data(df, database_filename):
    '''
    Save the dataframe to a database file of the specified name.
    The table name is 'messages'. The table is replaced if it exists.
    Inputs:
        df: (pandas dataframe) dataframe to be saved to the database
        database_file: (string) filepath to database file
    Returns: none
    '''
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('messages', engine, index=False, if_exists='replace')
    


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