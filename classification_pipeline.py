import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, roc_auc_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

# Import classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import xgboost as XGBClassifier

# Import the outlier removal function
from outlier import remove_outliers_iqr

# Load the data
print("Loading data...")
data = pd.read_csv('data.csv')

# Display basic information about the dataset
print("\nDataset Information:")
print(f"Number of samples: {data.shape[0]}")
print(f"Number of features: {data.shape[1]}")
print(f"Features: {data.columns.tolist()}")
print(f"Target distribution:\n{data['pc_class'].value_counts()}")

# Prepare X and y
# Exclude 'name' and 'country' columns and keep only numeric features
X = data.drop(['name', 'country', 'pc_class'], axis=1)
y = data['pc_class']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nTraining set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")

# Apply outlier removal to X_train
print("\nRemoving outliers from training data...")
X_train_no_outliers = remove_outliers_iqr(X_train)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_no_outliers)
X_test_scaled = scaler.transform(X_test)

# Apply SMOTE for oversampling the minority classes
print("\nApplying SMOTE to balance the classes...")
smote = SMOTE(random_state=42)
X_train_scaled, y_train = smote.fit_resample(X_train_scaled, y_train)
print(f"After SMOTE - Training data shape: {X_train_scaled.shape}")
print(f"After SMOTE - Target distribution:\n{pd.Series(y_train).value_counts()}")

# Define classifiers to evaluate - using a reduced set for efficiency
classifiers = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, class_weight={0: 2.5, 1: 1, 2: 1}),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight={0: 2.5, 1: 1, 2: 1}),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'XGBoost': XGBClassifier.XGBClassifier(n_estimators=100, random_state=42, learning_rate=0.1, use_label_encoder=False, eval_metric='mlogloss'),
    'Naive Bayes': GaussianNB()
}

# Train and evaluate each classifier
results = {}
metrics_results = {}
print("\nTraining and evaluating classifiers...")

for name, clf in classifiers.items():
    print(f"\nTraining {name}...")

    # Train the classifier
    if name in ['Gradient Boosting', 'XGBoost']:
        # Create sample weights based on class weights for Gradient Boosting and XGBoost
        class_weights = {0: 2.5, 1: 1, 2: 1}
        sample_weights = np.array([class_weights[y] for y in y_train])
        clf.fit(X_train_scaled, y_train, sample_weight=sample_weights)
    else:
        clf.fit(X_train_scaled, y_train)

    # Make predictions
    y_pred = clf.predict(X_test_scaled)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    # Store all metrics
    results[name] = accuracy
    metrics_results[name] = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1
    }

    # Print results
    print(f"{name} Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")
    print(f"Classification Report:\n{classification_report(y_test, y_pred)}")

    # We'll only generate visualizations for the best model later

# Find the best classifier
best_classifier = max(results, key=results.get)
print(f"\nBest classifier: {best_classifier} with accuracy: {results[best_classifier]:.4f}")

# Fine-tune the best classifier with more extensive parameter grids
print(f"\nFine-tuning {best_classifier}...")

if best_classifier == 'Logistic Regression':
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'solver': ['liblinear', 'lbfgs', 'saga'],
        'penalty': ['l1', 'l2', 'elasticnet', None],
        'max_iter': [1000, 2000]
    }
    best_clf = LogisticRegression(random_state=42, class_weight={0: 2.5, 1: 1, 2: 1})
elif best_classifier == 'Random Forest':
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }
    best_clf = RandomForestClassifier(random_state=42, class_weight={0: 2.5, 1: 1, 2: 1})
elif best_classifier == 'Gradient Boosting':
    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 5],
        'subsample': [0.8, 1.0]
    }
    best_clf = GradientBoostingClassifier(random_state=42)
elif best_classifier == 'XGBoost':
    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'min_child_weight': [1, 3, 5],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        'gamma': [0, 0.1, 0.2]
    }
    best_clf = XGBClassifier.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss')
elif best_classifier == 'Naive Bayes':
    param_grid = {
        'var_smoothing': [1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6]
    }
    best_clf = GaussianNB()

# Perform grid search with reduced CV folds and parallel processing
# Note: X_train_scaled and y_train already have SMOTE applied
grid_search = GridSearchCV(best_clf, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1)
print(f"Starting grid search for {best_classifier}...")

# Apply sample weights for Gradient Boosting and XGBoost
if best_classifier in ['Gradient Boosting', 'XGBoost']:
    # Create sample weights based on class weights
    class_weights = {0: 2.5, 1: 1, 2: 1}
    sample_weights = np.array([class_weights[y] for y in y_train])
    grid_search.fit(X_train_scaled, y_train, sample_weight=sample_weights)
else:
    grid_search.fit(X_train_scaled, y_train)

# Get the best parameters
best_params = grid_search.best_params_
print(f"Best parameters: {best_params}")

# Save the hyperparameters to a CSV file
if best_classifier == 'Random Forest':
    # If Random Forest is the best classifier, use its best parameters
    rf_best_params = best_params
    # Convert the best parameters to a DataFrame
    params_df = pd.DataFrame([rf_best_params])
    # Save to CSV
    params_df.to_csv('random_forest_hyperparameters.csv', index=False)
    print("\nRandom Forest hyperparameters saved to 'random_forest_hyperparameters.csv'")
else:
    # If Random Forest is not the best classifier, perform a separate grid search for Random Forest
    print("\nPerforming grid search for Random Forest to save its hyperparameters...")
    rf_param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }
    rf_clf = RandomForestClassifier(random_state=42, class_weight={0: 2.5, 1: 1, 2: 1})
    rf_grid_search = GridSearchCV(rf_clf, rf_param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1)
    rf_grid_search.fit(X_train_scaled, y_train)
    rf_best_params = rf_grid_search.best_params_
    print(f"Random Forest best parameters: {rf_best_params}")
    # Convert the best parameters to a DataFrame
    params_df = pd.DataFrame([rf_best_params])
    # Save to CSV
    params_df.to_csv('random_forest_hyperparameters.csv', index=False)
    print("\nRandom Forest hyperparameters saved to 'random_forest_hyperparameters.csv'")

# Train the model with the best parameters
best_model = grid_search.best_estimator_

# Apply sample weights for Gradient Boosting and XGBoost
if best_classifier in ['Gradient Boosting', 'XGBoost']:
    # Create sample weights based on class weights
    class_weights = {0: 2.5, 1: 1, 2: 1}
    sample_weights = np.array([class_weights[y] for y in y_train])
    best_model.fit(X_train_scaled, y_train, sample_weight=sample_weights)
else:
    best_model.fit(X_train_scaled, y_train)

# Evaluate the fine-tuned model with multiple metrics
y_pred = best_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"\nFine-tuned {best_classifier} Metrics:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Classification Report:\n{classification_report(y_test, y_pred)}")

# Save predictions to CSV file
# Get the indices of the test set
test_indices = X_test.index
# Create a DataFrame with predictions
predictions_df = pd.DataFrame({
    'name': data.loc[test_indices, 'name'],
    'country': data.loc[test_indices, 'country'],
    'actual_class': y_test,
    'predicted_class': y_pred
})
# Save to CSV
predictions_df.to_csv('predictions.csv', index=False)
print("\nPredictions saved to 'predictions.csv'")

# Store metrics for the fine-tuned model
metrics_results[f"Fine-tuned {best_classifier}"] = {
    'Accuracy': accuracy,
    'Precision': precision,
    'Recall': recall,
    'F1 Score': f1
}

# Generate visualizations only for the best model

# Plot confusion matrix for the fine-tuned model with RdBu theme
# plt.figure(figsize=(8, 6))
# cm = confusion_matrix(y_test, y_pred)
# sns.heatmap(cm, annot=True, fmt='d', cmap='RdBu', xticklabels=sorted(y.unique()), yticklabels=sorted(y.unique()))
# plt.title(f'Confusion Matrix - Fine-tuned {best_classifier}')
# plt.ylabel('True Label')
# plt.xlabel('Predicted Label')
# plt.savefig(f'confusion_matrix_fine_tuned_{best_classifier.replace(" ", "_")}.png')
# plt.close()

# Generate ROC curve for the best model if it supports predict_proba
# if hasattr(best_model, 'predict_proba'):
#     # For multiclass, we need to use one-vs-rest approach
#     # Get probability predictions
#     y_prob = best_model.predict_proba(X_test_scaled)
# 
#     # Plot ROC curve for each class
#     plt.figure(figsize=(10, 8))
# 
#     # Convert y_test to one-hot encoding for ROC curve calculation
#     from sklearn.preprocessing import label_binarize
#     classes = sorted(y.unique())
#     y_test_bin = label_binarize(y_test, classes=classes)
# 
#     for i, class_label in enumerate(classes):
#         fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
#         roc_auc = auc(fpr, tpr)
#         plt.plot(fpr, tpr, lw=2, label=f'Class {class_label} (AUC = {roc_auc:.2f})')
# 
#     plt.plot([0, 1], [0, 1], 'k--', lw=2)
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title(f'ROC Curve - Fine-tuned {best_classifier}')
#     plt.legend(loc="lower right")
# 
#     # Set the color theme to match RdBu
#     plt.grid(True, linestyle='--', alpha=0.7)
#     plt.savefig(f'roc_curve_{best_classifier.replace(" ", "_")}.png')
#     plt.close()
#     print(f"ROC curve for Fine-tuned {best_classifier} saved as 'roc_curve_{best_classifier.replace(' ', '_')}.png'")

# Identify and visualize the most important features
if best_classifier in ['Random Forest', 'Gradient Boosting', 'XGBoost']:
    # These models have feature_importances_ attribute
    feature_importances = best_model.feature_importances_
    feature_names = X.columns

    # Create a DataFrame for better visualization
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importances
    }).sort_values('Importance', ascending=False)

#     # Plot feature importances using plotly express with RdBu theme
#     fig = px.bar(
#         importance_df.head(15),  # Show top 15 features
#         x='Importance',
#         y='Feature',
#         orientation='h',
#         title=f'Top 15 Most Important Features - {best_classifier}',
#         color='Importance',
#         color_continuous_scale='RdBu',
#         template='plotly_white'
#     )
# 
#     # Update layout
#     fig.update_layout(
#         xaxis_title='Importance',
#         yaxis_title='Feature',
#         yaxis=dict(autorange="reversed")  # To have the highest importance at the top
#     )
# 
#     # Save the figure
#     fig.write_image('feature_importance.png')
#     print("\nFeature importance chart saved as 'feature_importance.png'")

    # Create a graph showing variables that influence prediction errors
    print("\nAnalyzing features that influence prediction errors...")

    # Get the incorrectly predicted samples
    incorrect_indices = np.where(y_pred != y_test)[0]

    if len(incorrect_indices) > 0:
        # Extract the features of incorrectly predicted samples
        X_incorrect = X_test.iloc[incorrect_indices]

        # Calculate the mean absolute deviation from the overall mean for each feature
        overall_mean = X.mean()
        incorrect_mean = X_incorrect.mean()

        # Calculate the difference between incorrect samples and overall dataset
        feature_impact = pd.DataFrame({
            'Feature': feature_names,
            'Mean_Difference': abs(incorrect_mean - overall_mean)
        }).sort_values('Mean_Difference', ascending=False)

#         # Plot the features that most influence prediction errors
#         fig = px.bar(
#             feature_impact.head(15),  # Show top 15 features
#             x='Mean_Difference',
#             y='Feature',
#             orientation='h',
#             title='Variáveis que Mais Influenciam nos Erros de Previsão',
#             color='Mean_Difference',
#             color_continuous_scale='RdBu',
#             template='plotly_white'
#         )
# 
#         # Update layout
#         fig.update_layout(
#             xaxis_title='Mean Absolute Difference',
#             yaxis_title='Feature',
#             yaxis=dict(autorange="reversed")  # To have the highest impact at the top
#         )
# 
#         # Save the figure
#         fig.write_image('feature_impact_on_errors.png')
#         print("Feature impact on prediction errors chart saved as 'feature_impact_on_errors.png'")
    else:
        print("No prediction errors found to analyze feature impact.")

# Create a complete pipeline
pipeline = Pipeline([
    ('outlier_removal', remove_outliers_iqr),
    ('scaler', StandardScaler()),
    ('smote', SMOTE(random_state=42)),
    ('classifier', best_model)
])

# # Save the pipeline
# import joblib
# joblib.dump(pipeline, 'classification_pipeline.joblib')
# print("\nPipeline saved as 'classification_pipeline.joblib'")

# # Save the best model separately in pkl format
# joblib.dump(best_model, 'best_model.pkl')
# print(f"\nBest model ({best_classifier}) saved as 'best_model.pkl' with accuracy: {accuracy:.4f}")

# Also save all models with their accuracies for reference
all_models = {}
for name, clf in classifiers.items():
    all_models[name] = {'model': clf, 'accuracy': results[name]}
    # # Save each model in pkl format
    # joblib.dump(clf, f'{name.replace(" ", "_")}_model.pkl')
    # print(f"{name} model saved as '{name.replace(' ', '_')}_model.pkl' with accuracy: {results[name]:.4f}")

# Save the fine-tuned best model with its accuracy
all_models[f"Fine-tuned {best_classifier}"] = {'model': best_model, 'accuracy': accuracy}

# Find the model with the highest accuracy across all models (including fine-tuned)
best_model_name = max(all_models, key=lambda x: all_models[x]['accuracy'])
best_overall_model = all_models[best_model_name]['model']
best_accuracy = all_models[best_model_name]['accuracy']

# # Save the best overall model based on accuracy
# joblib.dump(best_overall_model, 'best_model_by_accuracy.pkl')
# print(f"\nBest model by accuracy ({best_model_name}) saved as 'best_model_by_accuracy.pkl' with accuracy: {best_accuracy:.4f}")

# Create a comparison chart with multiple metrics for all models
print("\nCreating model comparison chart with multiple metrics...")

# Prepare data for the comparison chart
model_names = list(metrics_results.keys())
metrics_data = []

for metric in ['Accuracy', 'Precision', 'Recall', 'F1 Score']:
    for model in model_names:
        metrics_data.append({
            'Model': model,
            'Metric': metric,
            'Value': metrics_results[model][metric]
        })

metrics_df = pd.DataFrame(metrics_data)

# Save metrics to CSV file
metrics_wide_df = pd.DataFrame()
for model in model_names:
    model_metrics = pd.DataFrame([metrics_results[model]], index=[model])
    metrics_wide_df = pd.concat([metrics_wide_df, model_metrics])

# Reset index to make 'Model' a column
metrics_wide_df.reset_index(inplace=True)
metrics_wide_df.rename(columns={'index': 'Model'}, inplace=True)

# # Save to CSV
# metrics_wide_df.to_csv('model_metrics.csv', index=False)
# print("\nModel metrics saved to 'model_metrics.csv'")

# # Create the bar chart with plotly express using RdBu theme and white background
# fig = px.bar(
#     metrics_df, 
#     x='Model', 
#     y='Value', 
#     color='Metric',
#     barmode='group',
#     title='Model Comparison - Multiple Metrics',
#     color_discrete_sequence=px.colors.diverging.RdBu,
#     template='plotly_white'
# )
# 
# # Update layout for better appearance
# fig.update_layout(
#     xaxis_title='Model',
#     yaxis_title='Score',
#     yaxis=dict(range=[0, 1]),
#     legend_title='Metric'
# )
# 
# # Save the figure as PNG
# fig.write_image('model_comparison.png')
# print("Model comparison chart saved as 'model_comparison.png'")

# Create visualizations for growth potential by country and continent

# Define a mapping of countries to continents (focusing on Americas)
continent_mapping = {
    # North America
    'United States': 'North America',
    'Canada': 'North America',
    'Mexico': 'North America',

    # Central America
    'Belize': 'Central America',
    'Costa Rica': 'Central America',
    'El Salvador': 'Central America',
    'Guatemala': 'Central America',
    'Honduras': 'Central America',
    'Nicaragua': 'Central America',
    'Panama': 'Central America',

    # South America
    'Argentina': 'South America',
    'Bolivia': 'South America',
    'Brazil': 'South America',
    'Chile': 'South America',
    'Colombia': 'South America',
    'Ecuador': 'South America',
    'Guyana': 'South America',
    'Paraguay': 'South America',
    'Peru': 'South America',
    'Suriname': 'South America',
    'Uruguay': 'South America',
    'Venezuela': 'South America',

    # Default for other countries
    'default': 'Other'
}

# Add continent column to predictions DataFrame
predictions_df['continent'] = predictions_df['country'].map(lambda x: continent_mapping.get(x, continent_mapping['default']))

# Create a DataFrame with growth potential by country
# We'll consider class 2 as high growth potential, class 1 as medium, and class 0 as low
predictions_df['growth_potential'] = predictions_df['predicted_class'].map({2: 'High', 1: 'Medium', 0: 'Low'})

# Count companies by country and growth potential
country_growth = predictions_df.groupby(['country', 'growth_potential']).size().reset_index(name='count')

# Get top 10 countries by high growth potential
high_growth_countries = country_growth[country_growth['growth_potential'] == 'High'].sort_values('count', ascending=False).head(10)

# Create bar chart for top 10 countries by growth potential
fig_country_bar = px.bar(
    high_growth_countries,
    x='country',
    y='count',
    title='Top 10 Countries by High Growth Potential Companies',
    color='count',
    color_continuous_scale='RdBu',
    template='plotly_white'
)

# Update layout
fig_country_bar.update_layout(
    xaxis_title='Country',
    yaxis_title='Number of Companies',
    coloraxis_showscale=True
)

# Save the figure as PNG
fig_country_bar.write_image('top10_countries_growth_potential_bar.png')
print("\nTop 10 countries by growth potential bar chart saved as 'top10_countries_growth_potential_bar.png'")

# Create choropleth map for countries by growth potential
# Aggregate data by country
country_total = predictions_df.groupby('country').size().reset_index(name='total')
country_high_growth = predictions_df[predictions_df['growth_potential'] == 'High'].groupby('country').size().reset_index(name='high_growth')
country_map_data = pd.merge(country_total, country_high_growth, on='country', how='left')
country_map_data['high_growth'] = country_map_data['high_growth'].fillna(0)
country_map_data['high_growth_percentage'] = (country_map_data['high_growth'] / country_map_data['total']) * 100

# Create choropleth map
fig_country_map = px.choropleth(
    country_map_data,
    locations='country',
    locationmode='country names',
    color='high_growth_percentage',
    hover_name='country',
    color_continuous_scale='RdBu',
    title='Percentage of High Growth Potential Companies by Country',
    template='plotly_white'
)

# Update layout
fig_country_map.update_layout(
    geo=dict(
        showframe=False,
        showcoastlines=True,
        projection_type='equirectangular'
    )
)

# Save the figure as PNG
fig_country_map.write_image('countries_growth_potential_map.png')
print("\nCountries by growth potential map saved as 'countries_growth_potential_map.png'")

# Create visualizations for growth potential by continent (Americas)
# Filter for Americas continents
americas_data = predictions_df[predictions_df['continent'].isin(['North America', 'Central America', 'South America'])]

# Count companies by continent and growth potential
continent_growth = americas_data.groupby(['continent', 'growth_potential']).size().reset_index(name='count')

# Create bar chart for Americas continents by growth potential
fig_continent_bar = px.bar(
    continent_growth,
    x='continent',
    y='count',
    color='growth_potential',
    title='Growth Potential of Companies by American Continent',
    barmode='group',
    color_discrete_sequence=px.colors.diverging.RdBu,
    template='plotly_white'
)

# Update layout
fig_continent_bar.update_layout(
    xaxis_title='Continent',
    yaxis_title='Number of Companies',
    legend_title='Growth Potential'
)

# Save the figure as PNG
fig_continent_bar.write_image('americas_growth_potential_bar.png')
print("\nAmericas growth potential bar chart saved as 'americas_growth_potential_bar.png'")

# Create choropleth map for Americas continents by growth potential
# Aggregate data by continent
continent_total = americas_data.groupby('continent').size().reset_index(name='total')
continent_high_growth = americas_data[americas_data['growth_potential'] == 'High'].groupby('continent').size().reset_index(name='high_growth')
continent_map_data = pd.merge(continent_total, continent_high_growth, on='continent', how='left')
continent_map_data['high_growth'] = continent_map_data['high_growth'].fillna(0)
continent_map_data['high_growth_percentage'] = (continent_map_data['high_growth'] / continent_map_data['total']) * 100

# Create choropleth map for Americas
# For continents, we'll create a custom map using a different approach
# Create a DataFrame with ISO codes for each continent region
continent_iso = pd.DataFrame({
    'continent': ['North America', 'Central America', 'South America'],
    'iso_alpha': ['USA', 'MEX', 'BRA']  # Representative countries for each region
})

# Merge with our data
continent_map_data = pd.merge(continent_map_data, continent_iso, on='continent', how='left')

fig_continent_map = px.choropleth(
    continent_map_data,
    locations='iso_alpha',
    locationmode='ISO-3',
    color='high_growth_percentage',
    hover_name='continent',
    color_continuous_scale='RdBu',
    title='Percentage of High Growth Potential Companies by American Continent',
    template='plotly_white',
    scope='world'  # Using world scope but will focus on Americas
)

# Update layout
fig_continent_map.update_layout(
    geo=dict(
        showframe=False,
        showcoastlines=True,
        projection_type='equirectangular'
    )
)

# Save the figure as PNG
fig_continent_map.write_image('americas_growth_potential_map.png')
print("\nAmericas growth potential map saved as 'americas_growth_potential_map.png'")

# List of required libraries for these visualizations
required_libraries = [
    'pandas',
    'numpy',
    'plotly',
    'plotly_express',
    'scikit-learn',
    'matplotlib',
    'seaborn',
    'imblearn',
    'xgboost',
    'kaleido'  # Required for saving plotly figures as PNG
]

print("\nRequired libraries for these visualizations:")
for lib in required_libraries:
    print(f"- {lib}")

print("\nClassification pipeline completed successfully!")
