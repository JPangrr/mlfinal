import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score

np.random.seed(42)

try:
    df = pd.read_csv('realistic_ocean_climate_dataset.csv')
    print("Dataset loaded successfully with shape: {df.shape}")
    print("\nFirst few rows of the dataset:")
    print(df.head())
except FileNotFoundError:
    print("Dataset file not found. Please update the file path.")
    print("\nCreating sample dataset for demonstration...")

    np.random.seed(42)
    n_samples = 1000
    dates = pd.date_range(start='2015-01-01', end='2023-12-31', period=n_samples)

    locations = ["Red Sea", "Great Barrier Reef", "Caribbean Sea", "Galápagos", "South China Sea", "Maldives", "Hawaiian Islands"]

    df.pd.DataFrame({
        'Date': dates,
        'Location': np.random.choice(locations, n_samples),
        'Latitude': np.random.uniform(-90, 90, n_samples),
        'Longitude': np.random.uniform(-180, 180, n_samples),
        'SST': np.random.uniform(20, 30, n_samples),
        'pH Level': np.random.uniform(7.5, 8.5, n_samples),
        'Bleaching Severity': np.random.choice(['None', 'Low', 'Medium', 'High'], n_samples),
        'Species Observed': np.random.poisson(100, n_samples),
        'Marine Heatwave': np.random.choice(["True", "False"], n_samples)
    })

    for col in df.columns[2:]:
        mask = np.random.random(n_samples) < 0.05
        if df[col].dtype != 'object':
            df.loc[mask, col] = np.nan
        else:
            df.loc[mask, col] = None

    print(f"Sample dataset created with shape: {df.shape}")
    print("\nFirst few rows of the sample dataset:")
    print(df.head())

print("\nData Information:")
print(df.info())

print("\nSummary Statistics:")
print(df.describe())

print("\nMissing Values Count:")
print(df.isnull().sum())


############################
# parker's profiling part
############################

print("\nColumn Value Counts (Bleaching Severity):")
print(df["Bleaching Severity"].value_counts())

df['Bleaching Severity'] = df['Bleaching Severity'].fillna('None') 
# apparently 'None' wasn't a string, so covert all na values to 'None'

encoding_severity = {
    'None': 0,
    'Low': 1,
    'Medium': 2,
    'High': 3
}

df['Bleaching Severity Encoded'] = df['Bleaching Severity'].map(encoding_severity)
print(df["Bleaching Severity Encoded"].value_counts())

############################
# end of parker's profiling part
############################


if 'date' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['date']):
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['season'] = (df['month'] & 12 + 3) // 3

numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
categorical_cols.remove('Bleaching Severity') # added this to remove original column

if 'date' in numerical_cols:
    numerical_cols.remove('date')


print("\nNumerical Columns:", numerical_cols)
print("Categorical Columns:", categorical_cols)

plt.figure(figsize=(12, 10))
correlation = df[numerical_cols].corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Matrix of Numerical Features")
plt.tight_layout()
plt.savefig('correlation_matrix.png')
plt.close()

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ]
)

# Ensure column names are stripped of whitespace
df.columns = df.columns.str.strip()

# Check if 'SST' exists before proceeding
if 'SST' in df.columns:
    regression_target = 'SST'

    X = df.drop(columns=[regression_target, 'date'] if 'date' in df.columns else [regression_target])
    y = df[regression_target]

    X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(X, y, test_size=0.2, random_state=42)

    linear_reg_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])

    linear_reg_pipeline.fit(X_reg_train, y_reg_train)
    y_pred_linear = linear_reg_pipeline.predict(X_reg_test)
    mse_linear = mean_squared_error(y_reg_test, y_pred_linear)
    rmse_linear = np.sqrt(mse_linear)
    r2_linear = r2_score(y_reg_test, y_pred_linear)

    print("\n\nLinear Regression Results for SST:")
    print(f"Mean Squared Error: {mse_linear:.4f}")
    print(f"Root Mean Squared Error: {rmse_linear:.4f}")
    print(f"R² Score: {r2_linear:.4f}")
else:
    print("Column 'SST' not found in the dataset.")
    print("Available columns:", df.columns.tolist())

if "Bleaching Severity" in df.columns:
    df['Bleaching_Category'] = pd.qcut(
        df['Bleaching Severity'].fillna(df['Bleaching Severity'].median()),
        4,
        labels=["None", "Low", "Medium", "High"]
    )

    classification_target = 'Bleaching_Category'
    X_cls = df.drop(columns=[classification_target, "Bleaching Severity", 'date'] if 'date' in df.columns else [classification_target, "Bleaching Severity"])
    y_cls = df[classification_target]

    X_cls_train, X_cls_test, y_cls_train, y_cls_test = train_test_split(X_cls, y_cls, test_size=0.2, random_state=42, stratify=y_cls)

    log_reg_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
    ])
    log_reg_pipeline.fit(X_cls_train, y_cls_train)
    y_pred_log_reg = log_reg_pipeline.predict(X_cls_test)

    accuracy_log = accuracy_score(y_cls_test, y_pred_log_reg)

    print("\nLogistic Regression Results for {classification_target}:")
    print(f"Accuracy: {accuracy_log:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_cls_test, y_pred_log_reg))

    rf_cls_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    rf_cls_pipeline.fit(X_cls_train, y_cls_train)
    y_pred_rf_cls = rf_cls_pipeline.predict(X_cls_test)

    accuracy_rf_cls = accuracy_score(y_cls_test, y_pred_rf_cls)

    print("\nRandom Forest Classification Results for {classification_target}:")
    print(f"Accuracy: {accuracy_rf_cls:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_cls_test, y_pred_rf_cls))

if 'date' in df.columns and 'SST' in df.columns:
    print("\n\nTime Series Analysis for SST:")

    temp_time_series = df.groupby(df['date'].dt.to_period('M'))['SST'].mean().reset_index()
    temp_time_series['date'] = temp_time_series['date'].dt.to_timestamp()

    plt.figure(figsize=(14, 7))
    plt.plot(temp_time_series['date'], temp_time_series['SST'], marker='o', linestyle='-')
    plt.title("Average Sea Surface Temperature Over Time")
    plt.xlabel("Date")
    plt.ylabel("Average SST")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('sst_trend.png')
    plt.close()

    try:
        from sklearn.linear_model import LinearRegression

        first_date = temp_time_series['date'].min()
        temp_time_series['days_since_start'] = (temp_time_series['date'] - first_date).dt.days
        
        X_time = temp_time_series['days_since_start'].values.reshape(-1, 1)
        y_time = temp_time_series['SST'].values

        trend_model = LinearRegression()
        trend_model.fit(X_time, y_time)

        days_per_year = 365.25
        annual_change = trend_model.coef_[0] * days_per_year

        print(f"Temperature trend: {annual_change:.4f} °C per year")

        plt.figure(figsize=(14, 7))
        plt.scatter(temp_time_series['date'], temp_time_series['SST'], alpha=0.7)

        plt.plot(temp_time_series['date'], trend_model.predict(X_time), color='red', linewidth=2)

        plt.title(f"Sea Surface Temperature Trend: {annual_change:.4f} °C per year")
        plt.xlabel("Date")
        plt.ylabel("Average SST")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('temperature_trend_with_regression.png')
        plt.close()

    except Exception as e:
        print(f"Error in time series trend analysis: {e}")
    
print("\n\n=== BASELINE MODEL PERFORMANCE SUMMARY ===")

if "SST (°C)" in df.columns:
    print("\nRegression Task (Predicting SST):")
    print(f"Linear Regression RMSE: {rmse_linear:.4f}, R²: {r2_linear:.4f}")

if 'Bleaching Severity' in df.columns:
    print("\nClassification Task (Predicting Bleaching Severity):")
    print(f"Logistic Regression Accuracy: {accuracy_log:.4f}")
    print(f"Random Forest Accuracy: {accuracy_rf_cls:.4f}")

if "date" in df.columns and "SST" in df.columns:
    print(f"Time Series Analysis:")
    print(f"Temperature Trend: {annual_change:.4f} °C per year")

print("\n=== NEXT STEPS ===")
print("1. Feature engineering: Create lag features and seasonal indicators")
print("2. Try more advanced models: Gradient Boosting, Neural Networks")
print("3. Implement cross-validation for more robust evaluation")
print("4. Explore spatial patterns in the data")
print("5. Analyze correlations between climate variables and biodiversity")