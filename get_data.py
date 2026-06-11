import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import urllib.request
import shutil

import numpy as np
import pandas as pd # type: ignore
import tarfile
import zipfile
import os
import kagglehub # type: ignore
import argparse
import re

from dataset_profile import save_dataset_profile

DATA_DIR = "data/original"
LABEL_VERSIONS = list(range(10, 101, 10))
LCLD_SUBSAMPLE_SIZE = 10_000
WIDS_SUBSAMPLE_SIZE = 50_000


def dataset_csv_exists(dataset_name):
    return os.path.isfile(os.path.join(DATA_DIR, f"{dataset_name}.csv"))


def get_existing_max_version(prefix):
    max_version = None
    pattern = re.compile(rf"^{prefix}(\d+)\.csv$")
    for filename in os.listdir(DATA_DIR):
        match = pattern.match(filename)
        if match:
            version = int(match.group(1))
            max_version = version if max_version is None else max(max_version, version)
    return max_version


def get_max_target_labels(df):
    """Return the number of unique target labels in the last column."""
    return int(df.iloc[:, -1].nunique())


def filter_by_num_labels(df, num_labels):
    label_col = df.iloc[:, -1]
    unique_labels = np.sort(label_col.unique())
    selected = unique_labels[:num_labels]
    return df[label_col.isin(selected)].copy()


def list_csv_files(root_dir):
    csv_files = []
    for root, _, files in os.walk(root_dir):
        for filename in files:
            if filename.lower().endswith(".csv"):
                csv_files.append(os.path.join(root, filename))
    return csv_files


def read_csv_flexible(file_path):
    try:
        return pd.read_csv(file_path)
    except UnicodeDecodeError:
        return pd.read_csv(file_path, encoding="latin1")


def move_label_to_last(df, label_candidates):
    for label_col in label_candidates:
        if label_col in df.columns:
            df = df.copy()
            df[label_col] = df[label_col]
            df["Label"] = df.pop(label_col)
            # Some benchmark test splits can be unlabeled; keep only labeled rows.
            df = df[df["Label"].notna()].copy()
            return df
    return df


def select_train_test_files(csv_files):
    train_candidates = [
        f for f in csv_files if re.search(r"train|training", os.path.basename(f), re.IGNORECASE)
    ]
    test_candidates = [
        f for f in csv_files if re.search(r"test|testing|validation|valid|unlabeled", os.path.basename(f), re.IGNORECASE)
    ]

    if train_candidates and test_candidates:
        train_file = max(train_candidates, key=os.path.getsize)
        test_file = max(test_candidates, key=os.path.getsize)
        return train_file, test_file
    return None, None


def load_kaggle_dataframe(repo_ids, dataset_key, label_candidates):
    last_error = None
    for repo_id in repo_ids:
        try:
            path = kagglehub.dataset_download(repo_id)
            csv_files = list_csv_files(path)
            if not csv_files:
                raise FileNotFoundError(f"No CSV files found for {repo_id}")

            train_file, test_file = select_train_test_files(csv_files)
            if train_file and test_file:
                train_df = read_csv_flexible(train_file)
                test_df = read_csv_flexible(test_file)

                # Keep only shared columns to safely concatenate splits from different schemas.
                common_cols = [c for c in train_df.columns if c in test_df.columns]
                if not common_cols:
                    raise ValueError("Train/test files have no shared columns")

                train_df = train_df[common_cols]
                test_df = test_df[common_cols]
                df = pd.concat([train_df, test_df], ignore_index=True)
                print(
                    f"{dataset_key}: merged train/test from {os.path.basename(train_file)} and {os.path.basename(test_file)}"
                )
            else:
                selected_csv = max(csv_files, key=os.path.getsize)
                df = read_csv_flexible(selected_csv)
                print(f"{dataset_key}: using single file {os.path.basename(selected_csv)}")

            df = move_label_to_last(df, label_candidates)
            return df
        except Exception as exc:
            last_error = exc

    raise RuntimeError(
        f"Could not load {dataset_key} from configured Kaggle repositories: {repo_ids}. Last error: {last_error}"
    )

def load_openml_dataset(dataset_id):
    """Download a dataset from OpenML and return a DataFrame with Label as last column."""
    try:
        import openml
    except ImportError:
        raise ImportError("openml package required: pip install openml")
    dataset = openml.datasets.get_dataset(dataset_id, download_data=True)
    X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
    df = X.copy()
    df["Label"] = y
    return df


def download_file(url, compression, compressed_file, temp_dir_name):
    urllib.request.urlretrieve(url, compressed_file)

    if compression == 'tar':
        with tarfile.open(compressed_file, 'r:gz') as tar:
            tar.extractall(temp_dir_name, filter='fully_trusted')
        os.remove(compressed_file)

    elif compression == 'zip':
        with zipfile.ZipFile(compressed_file, 'r') as zip_ref:
            zip_ref.extractall(temp_dir_name)
        os.remove(compressed_file)



def download_data(dataset_name):
    os.makedirs(DATA_DIR, exist_ok=True)
    temp_dir = ''
    df = []
    single_version = None

    if dataset_name.startswith("purchases") and dataset_name != "purchases":
        suffix = dataset_name[len("purchases"):]
        if suffix.isdigit():
            single_version = int(suffix)
            dataset_name = "purchases"

    if dataset_name.startswith("texas"):
        raise ValueError("Texas dataset has been removed from this project.")

    # For all single-output datasets, skip if the final CSV already exists.
    if dataset_name not in ("purchases",) and dataset_csv_exists(dataset_name):
        print(f"{dataset_name} dataset already exists - skipping download")
        return

    if dataset_name == 'locations':
        url = 'https://github.com/privacytrustlab/datasets/raw/refs/heads/master/dataset_location.tgz'
        temp_dir = 'locations_dir'
        if not dataset_exists(temp_dir) or not dataset_csv_exists(dataset_name):
            download_file(url, 'tar', 'dataset_location.tgz', temp_dir)
            df = pd.read_csv(os.path.join(temp_dir, 'bangkok'), header=None)
            df['Label'] = df.pop(0) # move label to last column
            df = df.copy()

            filtered_locations = []
            for i in range(0, len(df)):
                if df.iloc[i, -1] <= 10:
                    filtered_locations.append(df.iloc[i].tolist())
            df = pd.DataFrame(filtered_locations)
        else:
            print("Locations dataset already exists — skipping download")


    elif dataset_name == 'purchases': # generates purchases10, purchases20, ..., purchases100
        url = 'https://github.com/privacytrustlab/datasets/raw/refs/heads/master/dataset_purchase.tgz'
        temp_dir = 'purchases_dir'
        if single_version is not None and dataset_csv_exists(f"purchases{single_version}"):
            print(f"purchases{single_version} already exists - skipping download")
        else:
            existing_max = get_existing_max_version("purchases")
            if single_version is None and existing_max is not None:
                expected_versions = sorted(set(LABEL_VERSIONS + [existing_max]))
                if all(dataset_csv_exists(f"purchases{v}") for v in expected_versions):
                    print("Purchases datasets already exist - skipping download")
                    return

            download_file(url, 'tar', 'dataset_purchase.tgz', temp_dir)
            df = pd.read_csv(os.path.join(temp_dir, 'dataset_purchase'), header=None)
            df['Label'] = df.pop(0) # move label to last column
            df = df.copy()

            max_labels = get_max_target_labels(df)
            print(f"Detected maximum number of labels for purchases: {max_labels}")

            if single_version is None:
                versions = [v for v in LABEL_VERSIONS if v <= max_labels]
                if max_labels not in versions:
                    versions.append(max_labels)
            else:
                versions = [min(single_version, max_labels)]

            for num_labels in versions:
                out_name = f"purchases{num_labels}"
                if dataset_csv_exists(out_name):
                    print(f"{out_name} already exists - skipping")
                    continue
                filtered_df = filter_by_num_labels(df, num_labels)
                save_dataset_profile(filtered_df, out_name)
                preprocess_data(filtered_df, out_name)


    elif dataset_name == 'student_performance': # (24, 53, 36, 40, 24)
        url = 'https://archive.ics.uci.edu/static/public/320/student+performance.zip'
        temp_dir = 'student_performance_dir'
        if not dataset_exists(temp_dir) or not dataset_csv_exists(dataset_name):
            download_file(url, 'zip', 'student+performance.zip', temp_dir)
            with zipfile.ZipFile(os.path.join(temp_dir, 'student.zip'), 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            df = pd.read_csv(os.path.join(temp_dir, 'student-por.csv'), delimiter=";")
            df.drop(['G1','G2'], axis=1, inplace=True) # Remove these columns since they are strongly correlated to target label

            # Replace scores with 3 classes (under average, average, above average)
            df.loc[df.G3 <= 9, 'G3'] = 1
            df.G3 = df.G3.replace([10,11,12,13],2)
            df.loc[df.G3 >= 14, 'G3'] = 3
        else:
            print("Student_performance dataset already exists — skipping download")


    elif dataset_name == 'dropout_success': # (12, 8, 13, 26, 2)
        url = 'https://archive.ics.uci.edu/static/public/697/predict+students+dropout+and+academic+success.zip'
        temp_dir = 'dropout_success_dir'
        if not dataset_exists(temp_dir) or not dataset_csv_exists(dataset_name):
            download_file(url, 'zip', 'predict+students+dropout+and+academic+success.zip', temp_dir)
            df = pd.read_csv(os.path.join(temp_dir, 'data.csv'), delimiter=";")
        else:
            print("Dropout_success dataset already exists — skipping download")
        

    elif dataset_name == 'credit_data': # (9, 27, 25, 26, 1)
        url = 'https://archive.ics.uci.edu/static/public/144/statlog+german+credit+data.zip'
        temp_dir = 'credit_data_dir'
        if not dataset_exists(temp_dir) or not dataset_csv_exists(dataset_name):
            download_file(url, 'zip', 'statlog+german+credit+data.zip', temp_dir)
            df = pd.read_csv(os.path.join(temp_dir, 'german.data'), delimiter=" ", header=None)
            df['Label'] = df.pop(0)
        else:
            print("Credit_data dataset already exists — skipping download")
        

    elif dataset_name == 'cirrhosis': # (12, 31, 25, 29, 10)
        url = 'https://archive.ics.uci.edu/static/public/878/cirrhosis+patient+survival+prediction+dataset-1.zip'
        temp_dir = 'cirrhosis_dir'
        if not dataset_exists(temp_dir) or not dataset_csv_exists(dataset_name):
            download_file(url, 'zip', 'cirrhosis+patient+survival+prediction+dataset-1.zip', temp_dir)
            df = pd.read_csv(os.path.join(temp_dir, 'cirrhosis.csv'))
            df['Label'] = df.pop('Status')
            df.drop(['ID', 'Age'], axis=1, inplace=True)
        else:
            print("Cirrhosis dataset already exists — skipping download")
        
    elif dataset_name == 'half_million': # (7, 15, 19, 34, 1)
        if not dataset_exists(temp_dir):
            path = kagglehub.dataset_download("anthonytherrien/half-a-million-lifestyle")
            df = pd.read_csv(os.path.join(path, 'user_data.csv'))
            df.drop(['First Name', 'Last Name'], axis=1, inplace=True)
            df = df[:len(df) // 30]
            print(df.columns)

            filtered_choices = []
            for i in range(0, len(df)):
                if df.iloc[i, -1] < 10:
                    filtered_choices.append(df.iloc[i].tolist())
            df = pd.DataFrame(filtered_choices)
            print(sorted(np.unique(df.iloc[:, -1])))
            save_dataset_profile(df, dataset_name)
            preprocess_data(df, dataset_name)
        else:
            print("Half_million dataset already exists — skipping download")


    elif dataset_name == 'cleveland_heart': # (16, 24, 19, 9, 10)
        if not dataset_exists(temp_dir):
            path = kagglehub.dataset_download("muneer228/uci-chd-ml-dataset")
            df = pd.read_csv(os.path.join(path, 'processed.cleveland.data'), header=None)
        else:
            print("Cleveland_heart dataset already exists — skipping download")


    elif dataset_name == 'switzerland_heart': # (61, 72, 58, 37, 1)
        url = 'https://archive.ics.uci.edu/static/public/45/heart+disease.zip'
        temp_dir = 'switzerland_heart_dir'
        if not dataset_exists(temp_dir) or not dataset_csv_exists(dataset_name):
            download_file(url, 'zip', 'heart+disease.zip', temp_dir)
            df = pd.read_csv(os.path.join(temp_dir, 'processed.switzerland.data'), header=None)
        else:
            print("Switzerland_heart dataset already exists — skipping download")

    elif dataset_name == 'surgery':
        df = load_kaggle_dataframe(
            repo_ids=["aahanmehra/surgery"],
            dataset_key=dataset_name,
            label_candidates=["Outcome", "outcome", "target", "Target", "label", "Label"],
        )

    elif dataset_name == 'lcld':
        # Try multiple IDs because this dataset appears under different Kaggle publishers.
        df = load_kaggle_dataframe(
            repo_ids=[
                "vukglov/lending-club-loan-data-cleared",
                "wordsforthewise/lending-club",
                "ethon0426/lending-club-2007-2020q3",
            ],
            dataset_key=dataset_name,
            label_candidates=["loan_status", "Loan_Status", "label", "Label", "target", "Target"],
        )
        if len(df) > LCLD_SUBSAMPLE_SIZE:
            df = df.sample(n=LCLD_SUBSAMPLE_SIZE, random_state=42).reset_index(drop=True)
            print(f"lcld: subsampled to {LCLD_SUBSAMPLE_SIZE} rows")

    elif dataset_name == 'url':
        df = load_kaggle_dataframe(
            repo_ids=["shashwatwork/web-page-phishing-detection-dataset"],
            dataset_key=dataset_name,
            label_candidates=["result", "Result", "label", "Label", "class", "Class", "status", "Status"],
        )

    elif dataset_name == 'us_stocks_financial':  # OpenML 46527 — binary classification, 4392 rows, 231 features
        df = load_openml_dataset(46527)

    elif dataset_name == 'strikes':  # OpenML 549 — regression (strike_volume), binned into 3 classes, 625 rows, 7 features
        df = load_openml_dataset(549)
        df["Label"] = pd.qcut(pd.to_numeric(df["Label"], errors="coerce"), q=3, labels=[0, 1, 2])

    elif dataset_name == 'aloi':  # OpenML 40906 — binary classification, 49999 rows, 28 features
        df = load_openml_dataset(40906)

    elif dataset_name == 'annthyroid':  # OpenML 40886 — binary classification, ~7200 rows, 6 features
        df = load_openml_dataset(40886)

    elif dataset_name == 'credit_rating':  # OpenML 46552 — 10-class classification, 2029 rows, 30 features
        df = load_openml_dataset(46552)

    elif dataset_name == 'wids':
        df = load_kaggle_dataframe(
            repo_ids=["sangeetha1213/wids2021-dataset"],
            dataset_key=dataset_name,
            label_candidates=["diabetes_mellitus", "target", "Target", "label", "Label"],
        )
        # First column in WiDS is an index-like identifier; remove it.
        if df.shape[1] > 1:
            df = df.iloc[:, 1:].copy()
        if len(df) > WIDS_SUBSAMPLE_SIZE:
            df = df.sample(n=WIDS_SUBSAMPLE_SIZE, random_state=42).reset_index(drop=True)
            print(f"wids: subsampled to {WIDS_SUBSAMPLE_SIZE} rows")

    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"Unknown dataset '{dataset_name}'. No download handler matched.")

    if dataset_name not in ('purchases',) and not dataset_name.startswith('half_million'):
        save_dataset_profile(df, dataset_name)
        preprocess_data(df, dataset_name)
    if temp_dir != '' and os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


def preprocess_data(df, out_file):
    # Replace NAs with mode
    df.replace('?', np.nan, inplace=True)
    if df.isna().sum().sum() > 0:
        df = df.fillna(df.mode().iloc[0])

    # Remove columns with no variance (i.e., one value only)
    count = 0
    cols_todrop = []
    for col in df.columns:
        unique_values = np.unique(df[col].dropna())
        if unique_values.size == 1:
            print("column ", col, " doesn't have variance")
            count += 1
            cols_todrop.append(col)
    df.drop(cols_todrop, axis=1, inplace=True)
    print(count, " columns without variance")


    # Remove duplicated rows
    duplicate_rows = df.duplicated()
    count_dup = 0
    for row in duplicate_rows:
        if row: count_dup += 1
    print("number of duplicate rows:", count_dup)
    df.drop_duplicates(inplace=True)

    # Convert only object or string columns to categorical codes
    for col in df.columns:
        if pd.api.types.is_string_dtype(df[col]):
            try:
                # Try converting to numeric
                df[col] = pd.to_numeric(df[col], errors="raise")
            except (ValueError, TypeError):
                # Fallback: encode as categorical codes
                df[col] = df[col].astype("category").cat.codes
                
    # Map labels to consecutive indices starting in 0 (for cases where labels aren't consecutive numbers and/or don't start in 0)
    print("labels before mapping: ", sorted(np.unique(df.iloc[:, -1])))
    unique_labels = np.sort(df.iloc[:, -1].unique())
    label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
    df.iloc[:, -1] = df.iloc[:, -1].map(label_mapping).astype("int8")
    print("labels after mapping: ", sorted(np.unique(df.iloc[:, -1])))

    # Save data to csv file
    df.to_csv(os.path.join(DATA_DIR, out_file + '.csv'), index=False, header=False)


def download_all_datasets():
    for dataset in ['locations', 'purchases10', 'student_performance',
                    'dropout_success', 'credit_data', 'cirrhosis',
                    'surgery', 'lcld', 'url', 'wids',
                    'us_stocks_financial', 'strikes', 'aloi',
                    'annthyroid', 'credit_rating']:
        print(f"Downloading {dataset}...")
        download_data(dataset)


def dataset_exists(path):
    return os.path.isdir(path) and len(os.listdir(path)) > 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="all",
                        help="Dataset name or 'all'")
    args = parser.parse_args()
    if args.dataset == "all":
        download_all_datasets()
    else:
        download_data(args.dataset)
