import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score

np.random.seed(42)

try:
    df = pd.read_csv('realistic_ocean_climate_dataset.csv')
    print("Dataset loaded successfully with shape: {df.shape}")
    print("\n\n\n======== PROFILING and SUMMARY STATISTICS ========\n")
    print("First few rows of the dataset:")
    print(df.head())
except FileNotFoundError:
    print("Dataset file not found. Please update the file path.")
    print("\nCreating sample dataset for demonstration...")

    np.random.seed(42)
    n_samples = 1000
    dates = pd.date_range(start='2015-01-01', end='2023-12-31', periods=n_samples)

    locations = ["Red Sea", "Great Barrier Reef", "Caribbean Sea", "Galápagos", "South China Sea", "Maldives", "Hawaiian Islands"]

    df = pd.DataFrame({
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

print("\nColumn Value Counts (Bleaching Severity):")
print(df["Bleaching Severity"].value_counts())

df['Bleaching Severity'] = df['Bleaching Severity'].fillna('None')

encoding_severity = {
    'None': 0,
    'Low': 1,
    'Medium': 2,
    'High': 3
}

df['Bleaching Severity Encoded'] = df['Bleaching Severity'].map(encoding_severity)

# After loading data and before feature engineering
# Convert Marine Heatwave column to boolean if it's string
if 'Marine Heatwave' in df.columns:
    if df['Marine Heatwave'].dtype == 'object':
        df['Marine Heatwave'] = df['Marine Heatwave'].map({'True': True, 'False': False})
    # Make sure it's numeric for the model
    df['Marine Heatwave'] = df['Marine Heatwave'].astype(int)
    print("\nMarine Heatwave column converted to numeric format")

# Fix the date column processing - uppercase D and fix the bit operation
if 'Date' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['Date']):
    df['Date'] = pd.to_datetime(df['Date'])
    df['year'] = df['Date'].dt.year
    df['month'] = df['Date'].dt.month
    df['day'] = df['Date'].dt.day
    df['season'] = ((df['month'] % 12 + 3) // 3)  # Fixed calculation

numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

# Remove columns we don't want to use as features
if 'Bleaching Severity' in categorical_cols:
    categorical_cols.remove('Bleaching Severity')
if 'Date' in numerical_cols:
    numerical_cols.remove('Date')
if 'date' in numerical_cols:  # Also check lowercase
    numerical_cols.remove('date')

# Now make sure to update the df with the dropped column
df = df.drop(columns='Bleaching Severity', errors='ignore')

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

# Ensure column names are stripped of whitespace
df.columns = df.columns.str.strip()

# Check if 'SST' exists before proceeding
if 'SST' in df.columns:
    regression_target = 'SST'
    X = df.drop(columns=[regression_target, 'Date'] if 'Date' in df.columns else [regression_target])
    y = df[regression_target]
    X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(X, y, test_size=0.2, random_state=42)

    numerical_cols_SST = ['Latitude', 'Longitude', 'pH Level', 'Species Observed', 
                           'Bleaching Severity Encoded', 'Marine Heatwave']
    categorical_cols_SST = ['Location']

    preprocessor_SST = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols_SST),
            ('cat', categorical_transformer, categorical_cols_SST)
        ]
    )

    linear_reg_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor_SST),
        ('regressor', LinearRegression())
    ])

    linear_reg_pipeline.fit(X_reg_train, y_reg_train)
    y_pred_linear = linear_reg_pipeline.predict(X_reg_test)
    mse_linear = mean_squared_error(y_reg_test, y_pred_linear)
    rmse_linear = np.sqrt(mse_linear)
    r2_linear = r2_score(y_reg_test, y_pred_linear)

    print("\n\n\n======== PREDICTION RESULTS: SST ========\n")
    print(f"Mean Squared Error: {mse_linear:.4f}")
    print(f"Root Mean Squared Error: {rmse_linear:.4f}")
    print(f"R² Score: {r2_linear:.4f}")
else:
    print("Column 'SST' not found in the dataset.")
    print("Available columns:", df.columns.tolist())

if 'Bleaching Severity Encoded' in df.columns:
    severity_map = {0: "None", 1: "Low", 2: "Medium", 3: "High"}
    df['Bleaching_Category'] = df['Bleaching Severity Encoded'].map(severity_map)

    classification_target = 'Bleaching_Category'
    X_cls = df.drop(columns=[classification_target, "Bleaching Severity Encoded", 'Date'] if 'Date' in df.columns else [classification_target, "Bleaching Severity"])
    y_cls = df[classification_target]

    X_cls_train, X_cls_test, y_cls_train, y_cls_test = train_test_split(X_cls, y_cls, test_size=0.2, random_state=42, stratify=y_cls)

    numerical_cols_severity = ['Latitude', 'Longitude', 'SST', 'pH Level', 
                               'Species Observed', 'Marine Heatwave']
    categorical_cols_severity = ['Location']

    preprocessor_severity = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols_severity),
            ('cat', categorical_transformer, categorical_cols_severity)
        ]
    )

    log_reg_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor_severity),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
    ])
    log_reg_pipeline.fit(X_cls_train, y_cls_train)
    y_pred_log_reg = log_reg_pipeline.predict(X_cls_test)
    
    accuracy_log = accuracy_score(y_cls_test, y_pred_log_reg)

    print("\n\n\n======== PREDICTION RESULTS: SST ========\n")
    print("\nLogistic Regression Results for {classification_target}:")
    print(f"Accuracy: {accuracy_log:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_cls_test, y_pred_log_reg))

    rf_cls_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor_severity),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    rf_cls_pipeline.fit(X_cls_train, y_cls_train)
    y_pred_rf_cls = rf_cls_pipeline.predict(X_cls_test)

    accuracy_rf_cls = accuracy_score(y_cls_test, y_pred_rf_cls)

    print("\nRandom Forest Classification Results for {classification_target}:")
    print(f"Accuracy: {accuracy_rf_cls:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_cls_test, y_pred_rf_cls))

if 'Date' in df.columns and 'SST' in df.columns:
    print("\n\nTime Series Analysis for SST:")

    temp_time_series = df.groupby(df['Date'].dt.to_period('M'))['SST'].mean().reset_index()
    temp_time_series['Date'] = temp_time_series['Date'].dt.to_timestamp()

    plt.figure(figsize=(14, 7))
    plt.plot(temp_time_series['Date'], temp_time_series['SST'], marker='o', linestyle='-')
    plt.title("Average Sea Surface Temperature Over Time")
    plt.xlabel("Date")
    plt.ylabel("Average SST")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('sst_trend.png')
    plt.close()

    try:
        from sklearn.linear_model import LinearRegression

        first_date = temp_time_series['Date'].min()
        temp_time_series['days_since_start'] = (temp_time_series['Date'] - first_date).dt.days
        
        X_time = temp_time_series['days_since_start'].values.reshape(-1, 1)
        y_time = temp_time_series['SST'].values

        trend_model = LinearRegression()
        trend_model.fit(X_time, y_time)

        days_per_year = 365.25
        annual_change = trend_model.coef_[0] * days_per_year

        print(f"Temperature trend: {annual_change:.4f} °C per year")

        plt.figure(figsize=(14, 7))
        plt.scatter(temp_time_series['Date'], temp_time_series['SST'], alpha=0.7)

        plt.plot(temp_time_series['Date'], trend_model.predict(X_time), color='red', linewidth=2)

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

if 'Bleaching Severity Encoded' in df.columns:
    print("\nClassification Task (Predicting Bleaching Severity):")
    print(f"Logistic Regression Accuracy: {accuracy_log:.4f}")
    print(f"Random Forest Accuracy: {accuracy_rf_cls:.4f}")

if "Date" in df.columns and "SST" in df.columns:
    print(f"Time Series Analysis:")
    print(f"Temperature Trend: {annual_change:.4f} °C per year")

print("\n=== NEXT STEPS ===")
print("1. Feature engineering: Create lag features and seasonal indicators")
print("2. Try more advanced models: Gradient Boosting, Neural Networks")
print("3. Implement cross-validation for more robust evaluation")
print("4. Explore spatial patterns in the data")
print("5. Analyze correlations between climate variables and biodiversity")

print("\n\n======== IMPLEMENTING NEXT STEPS ========\n")

# 1. Feature Engineering: Create lag features and seasonal indicators
print("\n--- 1. Feature Engineering: Lag Features and Seasonal Indicators ---")

if 'Date' in df.columns and 'SST' in df.columns:
    # Sort by date to ensure proper lag creation
    df = df.sort_values('Date')
    
    # Create lag features (1, 2, and 3 months)
    df_with_lags = df.copy()
    
    # Group by location for proper lagging
    for location in df['Location'].unique():
        location_mask = df['Location'] == location
        location_data = df[location_mask].copy()
        
        # Create lag features for SST and pH Level
        for lag in [1, 2, 3]:
            location_data[f'SST_lag_{lag}'] = location_data['SST'].shift(lag)
            location_data[f'pH_lag_{lag}'] = location_data['pH Level'].shift(lag)
        
        # Update the data in the main dataframe
        df_with_lags.loc[location_mask] = location_data
    
    # Create seasonal indicators using sin and cos transforms for cyclical features
    df_with_lags['month_sin'] = np.sin(2 * np.pi * df_with_lags['month'] / 12)
    df_with_lags['month_cos'] = np.cos(2 * np.pi * df_with_lags['month'] / 12)
    
    # Drop rows with NaN values from lag creation
    df_with_lags = df_with_lags.dropna()
    
    print(f"Added lag features and seasonal indicators. New shape: {df_with_lags.shape}")
    print("New features:", [col for col in df_with_lags.columns if col not in df.columns])
    
    # 2. Try more advanced models
    print("\n--- 2. Advanced Models: Gradient Boosting and Neural Networks ---")
    
    from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.neural_network import MLPRegressor, MLPClassifier
    
    # For regression (predicting SST)
    X_advanced = df_with_lags.drop(columns=['SST', 'Date'])
    y_advanced = df_with_lags['SST']
    
    # Add the new features to numerical cols
    advanced_numerical_cols = numerical_cols.copy()
    advanced_numerical_cols.extend([col for col in df_with_lags.columns 
                                   if col.startswith(('SST_lag', 'pH_lag', 'month_'))])
    
    # Remove target from features
    if 'SST' in advanced_numerical_cols:
        advanced_numerical_cols.remove('SST')
    
    # Create a preprocessor for advanced models
    advanced_preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, advanced_numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )
    
    # Split data for advanced models
    X_adv_train, X_adv_test, y_adv_train, y_adv_test = train_test_split(
        X_advanced, y_advanced, test_size=0.2, random_state=42
    )
    
    # 3. Implement cross-validation
    print("\n--- 3. Cross-Validation for Robust Evaluation ---")
    
    # Gradient Boosting Regressor with cross-validation
    gb_pipeline = Pipeline(steps=[
        ('preprocessor', advanced_preprocessor),
        ('regressor', GradientBoostingRegressor(n_estimators=100, random_state=42))
    ])
    
    cv_scores_gb = cross_val_score(
        gb_pipeline, X_advanced, y_advanced, 
        cv=5, scoring='neg_root_mean_squared_error'
    )
    
    print(f"Gradient Boosting Cross-Validation RMSE: {-cv_scores_gb.mean():.4f} ± {cv_scores_gb.std():.4f}")
    
    # Neural Network Regressor with improved convergence settings
    mlp_pipeline = Pipeline(steps=[
        ('preprocessor', advanced_preprocessor),
        ('regressor', MLPRegressor(
            hidden_layer_sizes=(100, 50),        # Keep your existing architecture
            max_iter=2000,                       # Increased from 1000 to 2000
            alpha=0.001,                         # Add regularization
            learning_rate='adaptive',            # Use adaptive learning rate
            early_stopping=True,                 # Enable early stopping
            validation_fraction=0.1,             # Use 10% as validation set
            n_iter_no_change=10,                 # Stop if no improvement for 10 iterations
            random_state=42                      # Keep consistent results
        ))
    ])
    
    cv_scores_mlp = cross_val_score(
        mlp_pipeline, X_advanced, y_advanced, 
        cv=5, scoring='neg_root_mean_squared_error'
    )
    
    print(f"Neural Network Cross-Validation RMSE: {-cv_scores_mlp.mean():.4f} ± {cv_scores_mlp.std():.4f}")
    
    # Train final models on all data
    gb_pipeline.fit(X_adv_train, y_adv_train)
    y_pred_gb = gb_pipeline.predict(X_adv_test)
    rmse_gb = np.sqrt(mean_squared_error(y_adv_test, y_pred_gb))
    r2_gb = r2_score(y_adv_test, y_pred_gb)
    
    mlp_pipeline.fit(X_adv_train, y_adv_train)
    y_pred_mlp = mlp_pipeline.predict(X_adv_test)
    rmse_mlp = np.sqrt(mean_squared_error(y_adv_test, y_pred_mlp))
    r2_mlp = r2_score(y_adv_test, y_pred_mlp)
    
    print("\nFinal Test Set Performance:")
    print(f"Gradient Boosting: RMSE = {rmse_gb:.4f}, R² = {r2_gb:.4f}")
    print(f"Neural Network: RMSE = {rmse_mlp:.4f}, R² = {r2_mlp:.4f}")
    print(f"Linear Regression: RMSE = {rmse_linear:.4f}, R² = {r2_linear:.4f}")
    
    # 4. Explore spatial patterns in the data
    print("\n--- 4. Spatial Patterns Analysis ---")
    
    plt.figure(figsize=(14, 10))
    
    # Calculate average SST by location
    location_sst = df.groupby('Location')['SST'].mean().sort_values(ascending=False)
    
    # Create a bar plot of SST by location
    sns.barplot(x=location_sst.index, y=location_sst.values)
    plt.title('Average Sea Surface Temperature by Location')
    plt.xlabel('Location')
    plt.ylabel('Average SST (°C)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('spatial_patterns_sst.png')
    plt.close()
    
    # Create a scatter plot of latitude vs. SST
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(df['Longitude'], df['Latitude'], c=df['SST'], 
                         cmap='coolwarm', alpha=0.6, s=50)
    plt.colorbar(scatter, label='Sea Surface Temperature (°C)')
    plt.title('Spatial Distribution of Sea Surface Temperature')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add location labels
    for location in df['Location'].unique():
        subset = df[df['Location'] == location]
        mean_lat = subset['Latitude'].mean()
        mean_lon = subset['Longitude'].mean()
        plt.annotate(location, (mean_lon, mean_lat), 
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('geographic_sst_distribution.png')
    plt.close()
    
    # 5. Analyze correlations between climate variables and biodiversity
    print("\n--- 5. Climate-Biodiversity Correlation Analysis ---")
    
    # Calculate correlations
    climate_biodiversity_cols = ['SST', 'pH Level', 'Bleaching Severity Encoded', 
                                'Species Observed', 'Marine Heatwave']
    climate_biodiversity_corr = df[climate_biodiversity_cols].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(climate_biodiversity_corr, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Between Climate Variables and Biodiversity')
    plt.tight_layout()
    plt.savefig('climate_biodiversity_correlation.png')
    plt.close()
    
    # Species vs Temperature scatter plot with trend line
    plt.figure(figsize=(10, 6))
    sns.regplot(x='SST', y='Species Observed', data=df, scatter_kws={'alpha':0.5})
    plt.title('Relationship Between Sea Surface Temperature and Species Diversity')
    plt.xlabel('Sea Surface Temperature (°C)')
    plt.ylabel('Number of Species Observed')
    plt.tight_layout()
    plt.savefig('temperature_biodiversity_relationship.png')
    plt.close()
    
    # Marine Heatwave impact on biodiversity (boxplot)
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Marine Heatwave', y='Species Observed', data=df)
    plt.title('Impact of Marine Heatwaves on Species Diversity')
    plt.xlabel('Marine Heatwave Present')
    plt.ylabel('Number of Species Observed')
    plt.tight_layout()
    plt.savefig('heatwave_biodiversity_impact.png')
    plt.close()
    
    print("\nAdvanced Analysis Complete!")
    print("Generated visualizations:")
    print("- spatial_patterns_sst.png - Average temperature by location")
    print("- geographic_sst_distribution.png - Geographic distribution of temperatures")
    print("- climate_biodiversity_correlation.png - Correlation matrix of climate and biodiversity")
    print("- temperature_biodiversity_relationship.png - Relationship between SST and species count")
    print("- heatwave_biodiversity_impact.png - Impact of heatwaves on biodiversity")
    
    # Final performance comparison
    models_comparison = {
        'Linear Regression': rmse_linear,
        'Gradient Boosting': rmse_gb,
        'Neural Network': rmse_mlp
    }
    
    print("\nModel Performance Comparison (RMSE, lower is better):")
    for model, rmse in sorted(models_comparison.items(), key=lambda x: x[1]):
        print(f"{model}: {rmse:.4f}")
    
    best_model = min(models_comparison.items(), key=lambda x: x[1])[0]
    print(f"\nBest performing model: {best_model}")
    
else:
    print("Cannot implement advanced analysis: Required 'Date' or 'SST' columns missing.")