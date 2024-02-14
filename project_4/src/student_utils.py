import pandas as pd
import numpy as np
import os
import tensorflow as tf
import random
import functools


####### STUDENTS FILL THIS OUT ######
#Question 3
def reduce_dimension_ndc(df, code_df):
    '''
    df: pandas dataframe, input dataset
    ndc_df: pandas dataframe, drug code dataset used for mapping in generic names
    return:
        df: pandas dataframe, output dataframe with joined generic drug name
    '''
    def get_generic_drug_name(ndc_code,code_df):
        if pd.isna(ndc_code):
          return np.nan
        else:
          sel  = code_df[code_df['NDC_Code'] == ndc_code]
          if sel.empty:
            return np.nan
          else:
            return sel.iloc[0][2]
               
    df['generic_drug_name'] = df['ndc_code'].apply(lambda x: get_generic_drug_name(x,code_df))
    return df

#Question 4
def select_first_encounter(df):
    '''
    df: pandas dataframe, dataframe with all encounters
    return:
        - first_encounter_df: pandas dataframe, dataframe with only the first encounter for a given patient
    '''
    first_encounter_df = df.iloc[df.groupby(['patient_nbr'])['encounter_id'].idxmin()]    
    return first_encounter_df


#Question 6
def patient_dataset_splitter(df, partition_key='patient_nbr', proportions = [60,20,20]):
    '''
    df: pandas dataframe, input dataset that will be split
    patient_key: string, column that is the patient id

    return:
     - train: pandas dataframe,
     - validation: pandas dataframe,
     - test: pandas dataframe,
    '''
    '''
    - Approximately 60%/20%/20%  train/validation/test split
    - Randomly sample different patients into each data partition
    - **IMPORTANT** Make sure that a patient's data is not in more than one partition, so that we can avoid possible data leakage.
    - Make sure that the total number of unique patients across the splits is equal to the total number of unique patients in the original dataset
    - Total number of rows in original dataset = sum of rows across all three dataset partitions
    '''
    # split on partition , we assume that the split on partition is also valid with the overall data 
    partition_key_list = df[partition_key].unique().tolist()
    partition_key_list  = random.sample(partition_key_list,len(partition_key_list)) 
    partition_key_list_length = len(partition_key_list)
    
    train_nbr = int(proportions[0]*partition_key_list_length/100)
    val_nbr =  int(proportions[1]*partition_key_list_length/100)
    tst_nbr = int(proportions[2]*partition_key_list_length/100)
    
    begin_idx = 0
    end_idx = train_nbr
    train_list = partition_key_list[:end_idx]
    
    begin_idx = end_idx
    end_idx = begin_idx + val_nbr
    
    val_list = partition_key_list[begin_idx:end_idx]
    begin_idx = end_idx
    
    tst_list = []
    
    if np.sum(proportions) == 100:
        tst_list = partition_key_list[begin_idx:]
    else:
        end_idx  = begin_idx + test_nbr
        tst_list = partition_key_list[begin_idx:end_idx]
    
    train = df[df[partition_key].isin(train_list)]
    validation = df[df[partition_key].isin(val_list)]
    test = df[df[partition_key].isin(tst_list)]
    
    df_length = len(df)
    train_percent = int(len(train)/df_length*100)
    validation_percent = int(len(validation)/df_length*100)
    test_percent = int(len(test)/df_length*100)
        
    return train, train_percent, validation,validation_percent, test, test_percent

#Question 7

def create_tf_categorical_feature_cols(categorical_col_list,
                              vocab_dir='./diabetes_vocab/'):
    '''
    categorical_col_list: list, categorical field list that will be transformed with TF feature column
    vocab_dir: string, the path where the vocabulary text files are located
    return:
        output_tf_list: list of TF feature columns
    '''
    output_tf_list = []
    for c in categorical_col_list:
        vocab_file_path = os.path.join(vocab_dir,  c + "_vocab.txt")
        '''
        Which TF function allows you to read from a text file and create a categorical feature
        You can use a pattern like this below...
        tf_categorical_feature_column = tf.feature_column.......

        '''
        categorical_feature_column=tf_categorical_feature_column = tf.feature_column.categorical_column_with_vocabulary_file(
            c,
            vocab_file_path)
        tf_embedded_categorical_feature_column = tf.feature_column.embedding_column(categorical_feature_column,dimension=4)
        
        
        output_tf_list.append(tf_embedded_categorical_feature_column)
    return output_tf_list

#Question 8
def normalize_numeric_with_zscore(col, mean, std):
    '''
    This function can be used in conjunction with the tf feature column for normalization
    '''
    try:
        #print(type(col))
        ret = (col - mean) / std
        if not np.isfinite(ret).all():
            # If there are NaN or inf values, raise a warning
            # return col
            raise ValueError(f"NaN or inf values detected in the result. col: {col}, mean: {mean}, std: {std}")
        return ret    
    except ZeroDivisionError:
        ##print("Error: Division by zero.")
        ##return col
        raise ZeroDivisionError
    except Exception as e:
        #print(f"An error occurred: {e}")
        #return col
        raise e



def create_tf_numeric_feature(col, MEAN, STD, default_value=0):
    '''
    col: string, input numerical column name
    MEAN: the mean for the column in the training data
    STD: the standard deviation for the column in the training data
    default_value: the value that will be used for imputing the field

    return:
        tf_numeric_feature: tf feature column representation of the input field
    '''
    tf_numeric_feature = None
    try:
        normalizer = functools.partial(normalize_numeric_with_zscore,mean=MEAN,std=STD)     

        tf_numeric_feature=tf.feature_column.numeric_column(col,
                                                           default_value=default_value,
                                                           dtype=tf.dtypes.float32,
                                                           normalizer_fn=normalize)
    except Exception as e:                               
        tf_numeric_feature=tf.feature_column.numeric_column(col,default_value=default_value,dtype=tf.dtypes.float32)

    return tf_numeric_feature

#Question 9
def get_mean_std_from_preds(diabetes_yhat):
    '''
    diabetes_yhat: TF Probability prediction object
    '''
    m= None
    s= None
    if diabetes_yhat is not None:
        m = diabetes_yhat.mean()
        s = diabetes_yhat.stddev()
    return m, s

# Question 10
def get_student_binary_prediction(df, col, criteria):
    '''
    df: pandas dataframe prediction output dataframe
    col: str,  probability mean prediction field
    return:
        student_binary_prediction: pandas dataframe converting input to flattened numpy array and binary labels
    '''
    #Changed the interface to input the critera from outside the function  
    student_binary_prediction = df[col].apply(lambda x: 1 if x >= criteria else 0)
    return student_binary_prediction