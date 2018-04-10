import pandas as pd, numpy as np, time

def get_dataframe_from_csv(file_path, delimiter, types, columns):
    start = time.time()
    print("Reading file: " + file_path.split("/")[-1])
    df = pd.read_csv(file_path, sep=delimiter, dtype=types, usecols=columns)
    print("Finished reading file. " + str(round(time.time() - start)) + " seconds elapsed.")
    return df

def get_dataframe_from_excel(file_path):
    excel_object = pd.ExcelFile(file_path)
    df = excel_object.parse(excel_object.sheet_names[0])  # assuming only one sheet
    return df

def write_to_csv(df, path):
    pd.DataFrame.to_csv(df, path)

def print_dataframe_info(df, head_no = 5):
    print("--------Head--------")
    print(df.head(head_no))
    print("--------Info--------")
    print(df.info())
    print("--------Null--------")
    print(df.isnull().sum())