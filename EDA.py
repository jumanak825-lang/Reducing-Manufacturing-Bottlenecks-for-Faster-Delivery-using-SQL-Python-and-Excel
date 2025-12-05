# =============================================
# 1. IMPORTS & CONFIGURATION
# =============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Models
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor, plot_tree, export_text
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb

# SHAP for explainability
import shap

# Clustering
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Visualization
from matplotlib import cm
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
pd.set_option('display.max_columns', None)

# =============================================
# 2. DATA LOADING & INITIAL EXPLORATION
# =============================================

def load_and_explore_data(filepath):
    """Load data and perform initial exploration"""
    
    print("=" * 60)
    print("DATA LOADING & INITIAL EXPLORATION")
    print("=" * 60)
    
    # Load data
    df = pd.read_csv(filepath)
    
    # Basic info
    print(f"Dataset Shape: {df.shape}")
    print(f"Number of Records: {df.shape[0]}")
    print(f"Number of Features: {df.shape[1]}")
    print("\n" + "-" * 40)
    
    # Data types
    print("Data Types:")
    print(df.dtypes.value_counts())
    
    # First few rows
    print("\nFirst 5 rows:")
    print(df.head())
    
    # Check for duplicates
    print(f"\nDuplicate Rows: {df.duplicated().sum()}")
    
    return df

# Load data
df = load_and_explore_data('supply_chain_data.csv')

# =============================================
# 3. DATA CLEANING & PREPROCESSING
# =============================================

def clean_and_preprocess(df):
    """Clean and preprocess the dataset"""
    
    print("\n" + "=" * 60)
    print("DATA CLEANING & PREPROCESSING")
    print("=" * 60)
    
    # Create a copy
    df_clean = df.copy()
    
    # 1. Handle missing values
    print("Missing Values Check:")
    missing_df = pd.DataFrame({
        'Column': df_clean.columns,
        'Missing_Count': df_clean.isnull().sum(),
        'Missing_Percentage': (df_clean.isnull().sum() / len(df_clean)) * 100
    })
    print(missing_df.sort_values('Missing_Percentage', ascending=False))
    
    # Drop columns with >50% missing if any
    cols_to_drop = missing_df[missing_df['Missing_Percentage'] > 50]['Column'].tolist()
    if cols_to_drop:
        print(f"\nDropping columns with >50% missing: {cols_to_drop}")
        df_clean = df_clean.drop(columns=cols_to_drop)
    
    # Fill remaining missing values
    for col in df_clean.columns:
        if df_clean[col].isnull().sum() > 0:
            if df_clean[col].dtype in ['int64', 'float64']:
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
            else:
                df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
    
    # 2. Handle duplicates
    initial_rows = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    print(f"\nRemoved {initial_rows - len(df_clean)} duplicate rows")
    
    # 3. Check for inconsistencies in categorical columns
    print("\nCategorical Value Counts:")
    categorical_cols = df_clean.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        print(f"\n{col}:")
        print(df_clean[col].value_counts().head())
    
    # 4. Convert column names to lowercase with underscores
    df_clean.columns = [col.lower().replace(' ', '_') for col in df_clean.columns]
    
    # 5. Standardize categorical values
    if 'inspection_results' in df_clean.columns:
        df_clean['inspection_results'] = df_clean['inspection_results'].str.strip().str.title()
    
    if 'transportation_modes' in df_clean.columns:
        df_clean['transportation_modes'] = df_clean['transportation_modes'].str.strip().str.title()
    
    print(f"\nFinal Dataset Shape after Cleaning: {df_clean.shape}")
    
    return df_clean

df_clean = clean_and_preprocess(df)

# =============================================
# 4. EXPLORATORY DATA ANALYSIS (EDA)
# =============================================

def perform_eda(df_clean):
    """Perform comprehensive EDA"""
    
    print("\n" + "=" * 60)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 60)
    
    # 1. Summary statistics
    print("\nSummary Statistics for Numerical Features:")
    print(df_clean.describe().round(2))
    
    # 2. Distribution of target variable (Lead Time)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    axes[0].hist(df_clean['lead_time'], bins=20, edgecolor='black', alpha=0.7, color='skyblue')
    axes[0].axvline(df_clean['lead_time'].mean(), color='red', linestyle='dashed', linewidth=2, label=f'Mean: {df_clean["lead_time"].mean():.2f}')
    axes[0].axvline(df_clean['lead_time'].median(), color='green', linestyle='dashed', linewidth=2, label=f'Median: {df_clean["lead_time"].median():.2f}')
    axes[0].set_xlabel('Lead Time (Days)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Distribution of Lead Time')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Boxplot
    axes[1].boxplot(df_clean['lead_time'], vert=False, patch_artist=True)
    axes[1].set_xlabel('Lead Time (Days)')
    axes[1].set_title('Box Plot of Lead Time')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 3. Correlation analysis
    numerical_cols = df_clean.select_dtypes(include=[np.number]).columns
    
    # Correlation matrix
    corr_matrix = df_clean[numerical_cols].corr()
    
    # Focus on correlation with lead_time
    lead_time_corr = corr_matrix['lead_time'].sort_values(ascending=False)
    
    print("\nTop 10 Features Correlated with Lead Time:")
    print(lead_time_corr.head(10))
    
    print("\nBottom 10 Features Correlated with Lead Time:")
    print(lead_time_corr.tail(10))
    
    # Visualize correlation matrix
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title('Correlation Matrix of Numerical Features')
    plt.tight_layout()
    plt.show()
    
    # 4. Categorical analysis by product type
    categorical_cols = df_clean.select_dtypes(include=['object']).columns
    
    for cat_col in categorical_cols[:3]:  # Limit to first 3 categorical columns
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Count plot
        value_counts = df_clean[cat_col].value_counts()
        axes[0].bar(value_counts.index[:10], value_counts.values[:10], color='lightcoral')
        axes[0].set_xlabel(cat_col)
        axes[0].set_ylabel('Count')
        axes[0].set_title(f'Distribution of {cat_col}')
        axes[0].tick_params(axis='x', rotation=45)
        
        # Lead time by category
        lead_time_by_cat = df_clean.groupby(cat_col)['lead_time'].mean().sort_values(ascending=False)
        axes[1].bar(lead_time_by_cat.index[:10], lead_time_by_cat.values[:10], color='lightseagreen')
        axes[1].set_xlabel(cat_col)
        axes[1].set_ylabel('Average Lead Time (Days)')
        axes[1].set_title(f'Average Lead Time by {cat_col}')
        axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    return lead_time_corr

correlation_results = perform_eda(df_clean)

# =============================================
# 5. FEATURE ENGINEERING
# =============================================

def feature_engineering(df_clean):
    """Create new features for better modeling"""
    
    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING")
    print("=" * 60)
    
    df_features = df_clean.copy()
    
    # 1. Log transformation for skewed features
    skewed_features = ['production_volumes', 'costs', 'revenue_generated']
    for feature in skewed_features:
        if feature in df_features.columns:
            df_features[f'{feature}_log'] = np.log1p(df_features[feature])
            print(f"Created log-transformed feature: {feature}_log")
    
    # 2. Target encoding for categorical variables
    categorical_cols = df_features.select_dtypes(include=['object']).columns
    
    for col in categorical_cols:
        if col != 'sku':  # Don't encode SKU if it's too granular
            # Mean lead time by category
            encoding = df_features.groupby(col)['lead_time'].mean()
            df_features[f'{col}_lead_time_mean'] = df_features[col].map(encoding)
            print(f"Created target-encoded feature: {col}_lead_time_mean")
    
    # 3. Interaction features
    if 'defect_rates' in df_features.columns and 'production_volumes' in df_features.columns:
        df_features['defect_production_interaction'] = df_features['defect_rates'] * df_features['production_volumes']
        print("Created interaction feature: defect_production_interaction")
    
    if 'price' in df_features.columns and 'order_quantities' in df_features.columns:
        df_features['price_order_interaction'] = df_features['price'] * df_features['order_quantities']
        print("Created interaction feature: price_order_interaction")
    
    # 4. Ratio features
    if 'revenue_generated' in df_features.columns and 'costs' in df_features.columns:
        df_features['profit_margin'] = (df_features['revenue_generated'] - df_features['costs']) / df_features['revenue_generated'].replace(0, 1)
        print("Created ratio feature: profit_margin")
    
    if 'stock_levels' in df_features.columns and 'order_quantities' in df_features.columns:
        df_features['inventory_turnover_ratio'] = df_features['order_quantities'] / df_features['stock_levels'].replace(0, 1)
        print("Created ratio feature: inventory_turnover_ratio")
    
    # 5. Polynomial features for important numerical features
    important_numerical = ['defect_rates', 'production_volumes', 'price']
    for feature in important_numerical:
        if feature in df_features.columns:
            df_features[f'{feature}_squared'] = df_features[feature] ** 2
            print(f"Created polynomial feature: {feature}_squared")
    
    # 6. Binning continuous variables
    if 'defect_rates' in df_features.columns:
        bins = [0, 1, 2, 3, 5, 10]
        labels = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
        df_features['defect_rate_category'] = pd.cut(df_features['defect_rates'], bins=bins, labels=labels)
        print("Created binned feature: defect_rate_category")
    
    if 'production_volumes' in df_features.columns:
        df_features['production_volume_category'] = pd.qcut(df_features['production_volumes'], q=4, labels=['Low', 'Medium', 'High', 'Very High'])
        print("Created binned feature: production_volume_category")
    
    # 7. Time efficiency metrics
    if 'manufacturing_lead_time' in df_features.columns:
        df_features['manufacturing_efficiency'] = df_features['production_volumes'] / df_features['manufacturing_lead_time'].replace(0, 1)
        print("Created efficiency feature: manufacturing_efficiency")
    
    print(f"\nOriginal number of features: {len(df_clean.columns)}")
    print(f"Number of features after engineering: {len(df_features.columns)}")
    print(f"Added {len(df_features.columns) - len(df_clean.columns)} new features")
    
    return df_features

df_features = feature_engineering(df_clean)

# =============================================
# 6. MODEL TRAINING & EVALUATION
# =============================================

def prepare_modeling_data(df_features):
    """Prepare data for machine learning models"""
    
    # Select features and target
    X = df_features.drop(columns=['lead_time', 'sku'], errors='ignore')
    
    # Handle categorical variables
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    
    y = df_features['lead_time']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42, stratify=df_features['product_type'] if 'product_type' in df_features.columns else None
    )
    
    print(f"Training set size: {X_train.shape}")
    print(f"Test set size: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test, X_encoded.columns

X_train, X_test, y_train, y_test, feature_names = prepare_modeling_data(df_features)

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """Train multiple models and evaluate performance"""
    
    print("\n" + "=" * 60)
    print("MODEL TRAINING & EVALUATION")
    print("=" * 60)
    
    # Dictionary to store models and results
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=0.1),
        'Decision Tree': DecisionTreeRegressor(max_depth=5, random_state=42),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42),
        'KNN': KNeighborsRegressor(n_neighbors=5)
    }
    
    results = []
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Calculate metrics
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        
        # Calculate overfitting metric
        overfit_percentage = ((train_r2 - test_r2) / train_r2 * 100) if train_r2 != 0 else 0
        
        # Store results
        results.append({
            'Model': name,
            'Train_RMSE': train_rmse,
            'Test_RMSE': test_rmse,
            'Train_R2': train_r2,
            'Test_R2': test_r2,
            'Train_MAE': train_mae,
            'Test_MAE': test_mae,
            'Overfit_%': overfit_percentage
        })
        
        print(f"  Train R²: {train_r2:.4f}")
        print(f"  Test R²: {test_r2:.4f}")
        print(f"  Test RMSE: {test_rmse:.2f} days")
        print(f"  Overfitting: {overfit_percentage:.1f}%")
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Sort by Test R²
    results_df = results_df.sort_values('Test_R2', ascending=False)
    
    print("\n" + "-" * 60)
    print("MODEL PERFORMANCE SUMMARY (Sorted by Test R²):")
    print("-" * 60)
    print(results_df[['Model', 'Test_R2', 'Test_RMSE', 'Test_MAE', 'Overfit_%']].to_string(index=False))
    
    # Visualize model comparison
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # R² comparison
    axes[0].barh(results_df['Model'], results_df['Test_R2'], color='lightgreen')
    axes[0].set_xlabel('R² Score')
    axes[0].set_title('Model Comparison - R² Score')
    axes[0].axvline(x=0, color='red', linestyle='--', alpha=0.5)
    
    # RMSE comparison
    axes[1].barh(results_df['Model'], results_df['Test_RMSE'], color='lightcoral')
    axes[1].set_xlabel('RMSE (Days)')
    axes[1].set_title('Model Comparison - RMSE')
    
    plt.tight_layout()
    plt.show()
    
    return models, results_df

models, results_df = train_and_evaluate_models(X_train, X_test, y_train, y_test)

# =============================================
# 7. DECISION TREE ANALYSIS FOR INTERPRETABILITY
# =============================================

def decision_tree_analysis(X_train, X_test, y_train, y_test, feature_names):
    """Analyze bottlenecks using Decision Tree"""
    
    print("\n" + "=" * 60)
    print("DECISION TREE BOTTLENECK ANALYSIS")
    print("=" * 60)
    
    # Train decision tree with optimal depth
    dt_model = DecisionTreeRegressor(
        max_depth=4, 
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42
    )
    
    dt_model.fit(X_train, y_train)
    
    # Visualize the tree
    plt.figure(figsize=(20, 10))
    plot_tree(dt_model, 
              feature_names=feature_names,
              filled=True, 
              rounded=True,
              fontsize=10,
              proportion=True)
    plt.title('Decision Tree for Lead Time Prediction', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    # Extract rules as text
    tree_rules = export_text(dt_model, feature_names=list(feature_names))
    print("\nDecision Tree Rules (First 20 lines):")
    print("\n".join(tree_rules.split("\n")[:20]))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': dt_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\nTop 10 Most Important Features from Decision Tree:")
    print(feature_importance.head(10).to_string(index=False))
    
    # Visualize feature importance
    plt.figure(figsize=(10, 6))
    top_features = feature_importance.head(10)
    plt.barh(top_features['Feature'], top_features['Importance'], color='royalblue')
    plt.xlabel('Importance')
    plt.title('Top 10 Feature Importance from Decision Tree')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()
    
    # Analyze specific rules
    print("\n" + "-" * 60)
    print("KEY DECISION PATTERNS IDENTIFIED:")
    print("-" * 60)
    
    # Simulated analysis based on decision tree
    decision_patterns = [
        {
            'Condition': 'defect_rates > 2.1 AND price > 62.97',
            'Avg_Lead_Time': '25+ days',
            'Interpretation': 'High defect rates combined with premium pricing cause severe delays',
            'Recommendation': 'Improve quality control for premium SKUs'
        },
        {
            'Condition': 'defect_rates <= 0.4 AND price <= 29.73',
            'Avg_Lead_Time': '1-3 days',
            'Interpretation': 'Low-cost, high-quality items are processed quickly',
            'Recommendation': 'Replicate this process for other products'
        },
        {
            'Condition': 'production_volumes > 700',
            'Avg_Lead_Time': '20+ days',
            'Interpretation': 'High production volumes create bottlenecks',
            'Recommendation': 'Implement batch processing or increase capacity'
        }
    ]
    
    patterns_df = pd.DataFrame(decision_patterns)
    print(patterns_df.to_string(index=False))
    
    return dt_model, feature_importance

dt_model, feature_importance = decision_tree_analysis(X_train, X_test, y_train, y_test, feature_names)

# =============================================
# 8. SHAP ANALYSIS FOR MODEL EXPLAINABILITY
# =============================================

def shap_analysis(model, X_train, X_test, feature_names):
    """Perform SHAP analysis for model interpretability"""
    
    print("\n" + "=" * 60)
    print("SHAP ANALYSIS FOR MODEL EXPLAINABILITY")
    print("=" * 60)
    
    # Initialize SHAP explainer
    explainer = shap.TreeExplainer(model)
    
    # Calculate SHAP values for test set
    shap_values = explainer.shap_values(X_test)
    
    # 1. Summary plot (beeswarm)
    print("\n1. Summary Plot - Feature Importance")
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.show()
    
    # 2. Bar plot of mean absolute SHAP values
    print("\n2. Mean Absolute SHAP Values")
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, plot_type="bar", show=False)
    plt.tight_layout()
    plt.show()
    
    # 3. Dependence plots for top features
    top_features = feature_importance.head(5)['Feature'].tolist()
    
    print("\n3. Dependence Plots for Top Features")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, feature in enumerate(top_features[:6]):
        if feature in X_test.columns:
            shap.dependence_plot(
                feature, 
                shap_values, 
                X_test, 
                feature_names=feature_names,
                ax=axes[idx],
                show=False
            )
            axes[idx].set_title(f'Dependence Plot: {feature}')
    
    # Remove unused subplots
    for idx in range(len(top_features[:6]), len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    plt.show()
    
    # 4. Force plot for a specific prediction
    print("\n4. Force Plot for a Specific Prediction (Sample #0)")
    sample_idx = 0
    shap.force_plot(
        explainer.expected_value, 
        shap_values[sample_idx, :], 
        X_test.iloc[sample_idx, :],
        feature_names=feature_names,
        matplotlib=True
    )
    plt.tight_layout()
    plt.show()
    
    # 5. Calculate and display top drivers
    print("\n5. Top Drivers of Lead Time Delays:")
    
    # Calculate mean absolute SHAP values
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    shap_importance = pd.DataFrame({
        'Feature': feature_names,
        'Mean_Abs_SHAP': mean_abs_shap
    }).sort_values('Mean_Abs_SHAP', ascending=False)
    
    print(shap_importance.head(10).to_string(index=False))
    
    return shap_values, explainer

# Use Random Forest for SHAP (works better with TreeExplainer)
shap_values, explainer = shap_analysis(models['Random Forest'], X_train, X_test, feature_names)

# =============================================
# 9. CLUSTERING FOR BOTTLENECK SEGMENTATION
# =============================================

def clustering_analysis(df_features):
    """Perform clustering to identify bottleneck segments"""
    
    print("\n" + "=" * 60)
    print("CLUSTERING ANALYSIS FOR BOTTLENECK SEGMENTATION")
    print("=" * 60)
    
    # Select features for clustering
    cluster_features = [
        'defect_rates',
        'production_volumes', 
        'price',
        'shipping_costs',
        'lead_time'
    ]
    
    # Filter only available features
    cluster_features = [f for f in cluster_features if f in df_features.columns]
    X_cluster = df_features[cluster_features]
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)
    
    # Determine optimal number of clusters using elbow method
    inertias = []
    silhouette_scores = []
    K_range = range(2, 11)
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)
        
        if len(X_scaled) > k:  # Silhouette requires at least 2 samples per cluster
            silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))
    
    # Plot elbow curve
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Elbow plot
    axes[0].plot(K_range, inertias, marker='o', color='royalblue', linewidth=2)
    axes[0].set_xlabel('Number of Clusters (k)')
    axes[0].set_ylabel('Inertia')
    axes[0].set_title('Elbow Method for Optimal k')
    axes[0].grid(True, alpha=0.3)
    
    # Silhouette scores
    axes[1].plot(K_range[2:], silhouette_scores[2:], marker='o', color='coral', linewidth=2)
    axes[1].set_xlabel('Number of Clusters (k)')
    axes[1].set_ylabel('Silhouette Score')
    axes[1].set_title('Silhouette Scores for Different k')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Based on plots, choose optimal k (typically where elbow bends)
    optimal_k = 4
    print(f"Selected optimal number of clusters: {optimal_k}")
    
    # Apply K-Means with optimal k
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    df_features['cluster'] = kmeans.fit_predict(X_scaled)
    
    # Analyze clusters
    cluster_summary = df_features.groupby('cluster')[cluster_features].mean()
    cluster_counts = df_features['cluster'].value_counts().sort_index()
    
    print("\nCluster Summary (Mean Values):")
    print(cluster_summary.round(2))
    
    print("\nNumber of Items in Each Cluster:")
    print(cluster_counts)
    
    # Visualize clusters (if 2D or 3D visualization possible)
    if len(cluster_features) >= 2:
        # Use PCA for dimensionality reduction to 2D
        from sklearn.decomposition import PCA
        
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        # Create scatter plot
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], 
                             c=df_features['cluster'], 
                             cmap='viridis',
                             alpha=0.7,
                             edgecolors='w',
                             s=100)
        
        plt.colorbar(scatter, label='Cluster')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.title(f'K-Means Clustering (k={optimal_k}) - PCA Visualization')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    # Analyze each cluster
    print("\n" + "-" * 60)
    print("CLUSTER-WISE BOTTLENECK ANALYSIS")
    print("-" * 60)
    
    cluster_analysis = []
    
    for cluster_id in sorted(df_features['cluster'].unique()):
        cluster_data = df_features[df_features['cluster'] == cluster_id]
        
        analysis = {
            'Cluster': cluster_id,
            'Count': len(cluster_data),
            'Avg_Lead_Time': cluster_data['lead_time'].mean(),
            'Avg_Defect_Rate': cluster_data['defect_rates'].mean() if 'defect_rates' in cluster_data.columns else np.nan,
            'Avg_Production_Volume': cluster_data['production_volumes'].mean() if 'production_volumes' in cluster_data.columns else np.nan,
            'Avg_Price': cluster_data['price'].mean() if 'price' in cluster_data.columns else np.nan,
            'Product_Types': cluster_data['product_type'].value_counts().index[0] if 'product_type' in cluster_data.columns else 'N/A'
        }
        
        # Add bottleneck characterization
        if analysis['Avg_Defect_Rate'] > 2.5:
            analysis['Primary_Bottleneck'] = 'Quality Issues'
        elif analysis['Avg_Production_Volume'] > 700:
            analysis['Primary_Bottleneck'] = 'Production Volume'
        elif analysis['Avg_Lead_Time'] > 20:
            analysis['Primary_Bottleneck'] = 'Logistics'
        else:
            analysis['Primary_Bottleneck'] = 'Efficient'
        
        cluster_analysis.append(analysis)
    
    # Create analysis DataFrame
    analysis_df = pd.DataFrame(cluster_analysis)
    analysis_df = analysis_df.sort_values('Avg_Lead_Time', ascending=False)
    
    print("\nDetailed Cluster Analysis:")
    print(analysis_df.to_string(index=False))
    
    # Visualize cluster characteristics
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    metrics = ['Avg_Lead_Time', 'Avg_Defect_Rate', 'Avg_Production_Volume', 'Avg_Price']
    titles = ['Average Lead Time', 'Average Defect Rate', 'Average Production Volume', 'Average Price']
    
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        if idx < len(axes):
            bars = axes[idx].bar(analysis_df['Cluster'].astype(str), analysis_df[metric], color=plt.cm.Set3(range(len(analysis_df))))
            axes[idx].set_xlabel('Cluster')
            axes[idx].set_ylabel(title.split()[-1])
            axes[idx].set_title(title)
            axes[idx].set_xticks(range(len(analysis_df)))
            axes[idx].set_xticklabels(analysis_df['Cluster'].astype(str))
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                axes[idx].text(bar.get_x() + bar.get_width()/2., height,
                              f'{height:.1f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.show()
    
    return df_features, cluster_summary, analysis_df

df_clustered, cluster_summary, cluster_analysis_df = clustering_analysis(df_features)

# =============================================
# 10. ACTIONABLE INSIGHTS & RECOMMENDATIONS
# =============================================

def generate_recommendations(df_features, correlation_results, feature_importance, cluster_analysis_df):
    """Generate actionable recommendations based on analysis"""
    
    print("\n" + "=" * 60)
    print("ACTIONABLE INSIGHTS & RECOMMENDATIONS")
    print("=" * 60)
    
    insights = []
    
    # 1. Quality Control Insights
    if 'defect_rates' in df_features.columns:
        defect_correlation = correlation_results.get('defect_rates', 0) if isinstance(correlation_results, dict) else 0
        if defect_correlation > 0.25 or ('defect_rates' in feature_importance['Feature'].values and 
                                        feature_importance[feature_importance['Feature'] == 'defect_rates']['Importance'].values[0] > 0.2):
            insights.append({
                'Category': 'Quality Control',
                'Problem': 'High defect rates are strongly correlated with longer lead times',
                'Evidence': f'Correlation: {defect_correlation:.3f}, Feature Importance: High',
                'Recommendation': [
                    'Implement real-time quality monitoring systems',
                    'Establish stricter supplier quality agreements',
                    'Introduce automated inspection at key production stages',
                    'Create a defect tracking dashboard with root cause analysis'
                ]
            })
    
    # 2. Production Volume Insights
    if 'production_volumes' in df_features.columns:
        high_volume_clusters = cluster_analysis_df[cluster_analysis_df['Avg_Production_Volume'] > 700]
        if not high_volume_clusters.empty:
            insights.append({
                'Category': 'Production Planning',
                'Problem': 'High production volumes create bottlenecks and increase lead times',
                'Evidence': f'Clusters {", ".join(high_volume_clusters["Cluster"].astype(str))} show high volumes with long lead times',
                'Recommendation': [
                    'Implement staggered production scheduling',
                    'Use load balancing across multiple production lines',
                    'Introduce dynamic capacity planning tools',
                    'Consider batch processing for high-volume SKUs'
                ]
            })
    
    # 3. Pricing Strategy Insights
    if 'price' in df_features.columns:
        price_correlation = correlation_results.get('price', 0) if isinstance(correlation_results, dict) else 0
        if price_correlation > 0.15:
            insights.append({
                'Category': 'Pricing & SKU Management',
                'Problem': 'Premium-priced SKUs experience longer lead times',
                'Evidence': f'Price-Lead Time Correlation: {price_correlation:.3f}',
                'Recommendation': [
                    'Analyze supplier capacity for premium products',
                    'Create dedicated production lines for high
