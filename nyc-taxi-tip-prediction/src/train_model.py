import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

def load_data(taxi_path: str, preds_path: str) -> pd.DataFrame:
    """
    Load and merge taxi trip data with predicted means.
    """
    df0 = pd.read_csv(taxi_path)
    preds = pd.read_csv(preds_path)
    # Assuming both share the same index as in the notebook
    df0 = df0.merge(preds, left_index=True, right_index=True)
    return df0


def engineer_features(df0: pd.DataFrame) -> pd.DataFrame:
    """
    Perform feature engineering and return a modeling-ready DataFrame.
    This mirrors the main steps in the notebook.
    """

    # Filter to credit card payments only
    df1 = df0[df0['payment_type'] == 1].copy()

    # Compute tip_percent and target variable 'generous'
    df1['tip_percent'] = (df1['tip_amount'] /
                          (df1['total_amount'] - df1['tip_amount'])).round(4)
    df1['generous'] = (df1['tip_percent'] >= 0.20).astype(int)

    # Convert pickup/dropoff to datetime
    df1['tpep_pickup_datetime'] = pd.to_datetime(
        df1['tpep_pickup_datetime'], errors='coerce'
    )
    df1['tpep_dropoff_datetime'] = pd.to_datetime(
        df1['tpep_dropoff_datetime'], errors='coerce'
    )

    # Day of week
    df1['day'] = df1['tpep_pickup_datetime'].dt.day_name().str.lower()

    # Time-of-day bins: am_rush, daytime, pm_rush, nighttime
    hours = df1['tpep_pickup_datetime'].dt.hour

    df1['am_rush'] = ((hours >= 6) & (hours < 10)).astype(int)
    df1['daytime'] = ((hours >= 10) & (hours < 16)).astype(int)
    df1['pm_rush'] = ((hours >= 16) & (hours < 20)).astype(int)
    df1['nighttime'] = (((hours >= 20) & (hours < 24)) |
                        ((hours >= 0) & (hours < 6))).astype(int)

    # Month abbreviation
    df1['month'] = df1['tpep_pickup_datetime'].dt.strftime('%b').str.lower()

    # Drop columns that won't be available at prediction time or are redundant
    drop_cols = [
        'Unnamed: 0',
        'tpep_pickup_datetime',
        'tpep_dropoff_datetime',
        'payment_type',
        'trip_distance',
        'store_and_fwd_flag',
        'fare_amount',
        'extra',
        'mta_tax',
        'tip_amount',
        'tolls_amount',
        'improvement_surcharge',
        'total_amount',
        'tip_percent',
    ]
    drop_cols = [c for c in drop_cols if c in df1.columns]
    df1 = df1.drop(columns=drop_cols)

    # Convert numeric categorical IDs to string
    for col in ['RatecodeID', 'PULocationID', 'DOLocationID', 'VendorID']:
        if col in df1.columns:
            df1[col] = df1[col].astype(str)

    # One-hot encoding
    df2 = pd.get_dummies(df1, drop_first=True)

    return df2


def train_random_forest(X_train, y_train) -> RandomForestClassifier:
    """
    Train a Random Forest model with reasonably strong hyperparameters.
    (These can be replaced with the best params found in the notebook.)
    """
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        max_samples=0.7,
        class_weight=None,
        random_state=27,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    return rf


def evaluate_model(model, X_test, y_test, model_name: str = "Model"):
    """
    Compute and print classification metrics and confusion matrix.
    """
    preds = model.predict(X_test)

    accuracy = accuracy_score(y_test, preds)
    precision = precision_score(y_test, preds)
    recall = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    cm = confusion_matrix(y_test, preds)

    print(f"=== {model_name} Evaluation ===")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)


def main():
    # Adjust these paths as needed
    taxi_path = "data/2017_Taxi.csv"
    preds_path = "data/predicted_means.csv"

    print("Loading data...")
    df0 = load_data(taxi_path, preds_path)

    print("Engineering features...")
    df2 = engineer_features(df0)

    print("Splitting train/test...")
    y = df2['generous']
    X = df2.drop(columns=['generous'])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=27
    )

    print("Training Random Forest...")
    rf = train_random_forest(X_train, y_train)

    print("Evaluating on test set...")
    evaluate_model(rf, X_test, y_test, model_name="Random Forest")


if __name__ == "__main__":
    main()