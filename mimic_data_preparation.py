import pandas as pd
import os
import re
from sklearn.model_selection import train_test_split


def preprocess_admissions(admissions_path):
    admissions = pd.read_csv(admissions_path)
    admissions['admittime'] = pd.to_datetime(
        admissions['admittime'], errors='coerce')
    admissions['dischtime'] = pd.to_datetime(
        admissions['dischtime'], errors='coerce')
    admissions['deathtime'] = pd.to_datetime(
        admissions['deathtime'], errors='coerce')
    admissions['LOS'] = (admissions['dischtime'] -
                         admissions['admittime']).dt.total_seconds() / (24 * 3600)
    admissions = admissions.dropna(subset=['hadm_id', 'subject_id'])
    return admissions


def preprocess_notes(notes_path):
    notes = pd.read_csv(notes_path)
    notes['chartdate'] = pd.to_datetime(notes['chartdate'], errors='coerce')
    notes['charttime'] = pd.to_datetime(notes['charttime'], errors='coerce')
    notes['TEXT'] = notes['TEXT'].apply(lambda x: re.sub(
        # Remove de-identified placeholders
        r'\[\*\*.*?\*\*\]', '', x))
    notes['TEXT'] = notes['TEXT'].str.replace(
        # Remove punctuation and lowercase
        r'[^\w\s]', '', regex=True).str.lower()
    notes['charttime'] = notes.apply(lambda row: row['chartdate'].replace(
        hour=23, minute=59, second=59)
        if pd.isna(row['charttime']) else row['charttime'], axis=1)
    return notes


def prepare_ftl_trans_data(admissions_path, notes_path, data_dir='data'):
    # Preprocess admissions and notes
    admissions = preprocess_admissions(admissions_path)
    notes = preprocess_notes(notes_path)

    # Merge admissions with notes on HADM_ID
    data = notes.merge(admissions[['hadm_id', 'hospital_expire_flag']],
                       left_on='HADM_ID', right_on='hadm_id', how='inner')

    # Rename columns to match FTL-Trans expected format
    data = data.rename(columns={
        'HADM_ID': 'Adm_ID',
        'ROW_ID': 'Note_ID',
        'chartdate': 'chartdate',
        'charttime': 'charttime',
        'TEXT': 'TEXT',
        'hospital_expire_flag': 'Label'
    })

    # Select only the required columns
    data = data[['Adm_ID', 'Note_ID', 'chartdate',
                 'charttime', 'TEXT', 'Label']]

    # Split data at the admission level to avoid leakage
    unique_adm = data['Adm_ID'].unique()
    train_adm, temp_adm = train_test_split(
        unique_adm, test_size=0.2, random_state=42)
    val_adm, test_adm = train_test_split(
        temp_adm, test_size=0.5, random_state=42)

    # Create train, validation, and test sets
    train = data[data['Adm_ID'].isin(train_adm)]
    val = data[data['Adm_ID'].isin(val_adm)]
    test = data[data['Adm_ID'].isin(test_adm)]

    # Create data directory if it doesn’t exist
    os.makedirs(data_dir, exist_ok=True)

    # Save the splits to the data/ directory
    train.to_csv(os.path.join(data_dir, 'train.csv'), index=False)
    val.to_csv(os.path.join(data_dir, 'val.csv'), index=False)
    test.to_csv(os.path.join(data_dir, 'test.csv'), index=False)

    print(
        f"Data has been successfully saved to the '{data_dir}/' directory with train.csv, val.csv, and test.csv.")


if __name__ == "__main__":
    admissions_path = 'mimic/ADMISSIONS.CSV'
    notes_path = 'mimic/NOTEEVENTS.CSV'
    prepare_ftl_trans_data(admissions_path, notes_path)
