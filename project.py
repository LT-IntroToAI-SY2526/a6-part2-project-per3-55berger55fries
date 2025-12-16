"""
Multivariable Linear Regression Project
Assignment 6 Part 3

Group Members:
- Lyle
- Aylish
- Miranda
- Elisabeth

Dataset: [Name of your dataset]
Predicting: [Popularity]
Features: [Duration, Danceability, Energy, Tempo]
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# TODO: Update this with your actual filename
DATA_FILE = 'spotify_dataset.csv'

def load_and_explore_data(filename):
    """
    Load your dataset and print basic information
    
    TODO:
    - Load the CSV file
    - Print the shape (rows, columns)
    - Print the first few rows
    - Print summary statistics
    - Check for missing values
    """
    # Your code here
    data = pd.read_csv(filename)
    print("=" * 70)
    print("LOADING AND EXPLORING DATA")
    print("=" * 70)
    print(f"\nFirst 5 rows:")
    print(data.head())
    print(f"\nDataset shape: {data.shape[0]} rows, {data.shape[1]} columns")
    print(f"\nBasic Statistics:")
    print(data.describe())
    print(f"\nColumn names: {list(data.columns)}")
    return data

def visualize_data(data):
    """
    Create visualizations to understand your data
    
    TODO:
    - Create scatter plots for each feature vs target
    - Save the figure
    - Identify which features look most important
    
    Args:
        data: your DataFrame
        feature_columns: list of feature column names
        target_column: name of target column
    """
    print("\n" + "=" * 70)
    print("VISUALIZING RELATIONSHIPS")
    print("=" * 70)
    
    # Your code here
    # Hint: Use subplots like in Part 2!
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Song Features vs Popularity', fontsize = 16, fontweight = 'bold')

    # first feature (duration)
    axes[0,0].scatter(data['Duration'], data['Popularity'], color = 'blue', alpha = 0.6)
    axes[0,0].set_xlabel('Duration')
    axes[0,0].set_ylabel('Popularity (%)')
    axes[0,0].set_title('Duration vs Popularity')
    axes[0,0].grid(True, alpha=0.3)
    
    # second feature (danceability)
    axes[0,1].scatter(data['Danceability'], data['Popularity'], color = 'green', alpha = 0.6)
    axes[0,1].set_xlabel('Danceability')
    axes[0,1].set_ylabel('Popularity (%)')
    axes[0,1].set_title('Danceability vs Popularity')
    axes[0,1].grid(True, alpha=0.3)

    # third feature (extra sauce)
    axes[1,0].scatter(data['Energy'], data['Popularity'], color = 'red', alpha = 0.6)
    axes[1,0].set_xlabel('Energy')
    axes[1,0].set_ylabel('Popularity (%)')
    axes[1,0].set_title('Energy vs Popularity')
    axes[1,0].grid(True, alpha=0.3)

    # fourth feature (extra cheese)
    axes[1,1].scatter(data['Tempo'], data['Popularity'], color = 'orange', alpha = 0.6)
    axes[1,1].set_xlabel('Tempo')
    axes[1,1].set_ylabel('Popularity (%)')
    axes[1,1].set_title('Tempo vs. Popularity')
    axes[1,1].grid(True, alpha = 0.3)

    plt.tight_layout()
    plt.savefig('feature_plots.png', dpi = 300, bbox_inches='tight')
    plt.show()

def prepare_and_split_data(data):
    """
    Prepare X and y, then split into train/test
    
    TODO:
    - Separate features (X) and target (y)
    - Split into train/test (80/20)
    - Print the sizes
    
    Args:
        data: your DataFrame
        feature_columns: list of feature column names
        target_column: name of target column
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    print("\n" + "=" * 70)
    print("PREPARING AND SPLITTING DATA")
    print("=" * 70)
    feature_columns = ['Duration', 'Danceability', 'Energy', 'Tempo']
    # Your code here
    X = data[feature_columns]
    y = data['Popularity']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)

    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train, feature_names):
    """
    Train the linear regression model
    
    TODO:
    - Create and train a LinearRegression model
    - Print the equation with all coefficients
    - Print feature importance (rank features by coefficient magnitude)
    
    Args:
        X_train: training features
        y_train: training target
        feature_names: list of feature names
        
    Returns:
        trained model
    """
    print("\n" + "=" * 70)
    print("TRAINING MODEL")
    print("=" * 70)
    # Your code here
    model = LinearRegression()
    model.fit(X_train, y_train)
    print(f"Intercept: {model.intercept_:.2f}")
    print(f"\nCoefficients:")
    for name, coef in zip(feature_names, model.coef_):
        print(f"  {name}: {coef:2f}")
    print(f"\nEquation")
    equation = f"Popularity = "
    for i, (name, coef) in enumerate(zip(feature_names, model.coef_)):
        if i == 0:
            equation += f"{coef:.2f} x {name}"
        else:
            equation += f" + ({coef:.2f}) x {name}"
    equation += f" + {model.intercept_:.2f}"
    print(equation)
    return model 


def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance
    
    TODO:
    - Make predictions on test set
    - Calculate R² score
    - Calculate RMSE
    - Print results clearly
    - Create a comparison table (first 10 examples)
    
    Args:
        model: trained model
        X_test: test features
        y_test: test target
        
    Returns:
        predictions
    """
    print("\n" + "=" * 70)
    print("EVALUATING MODEL")
    print("=" * 70)
    
    # Your code here
    predictions = model.predict(X_test)
    r2 = r2_score(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    print(f"\n=== Model Performance ===")
    print(f"R² Score: {r2:.4f}")
    print(f"  → Model explains {r2*100:.2f}% of price variation")
    print(f"\nRoot Mean Squared Error: ${rmse:.2f}")
    print(f"  → On average, predictions are off by ${rmse:.2f}")
    
    
   
    return predictions
    


def make_prediction(model, duration, danceability, energy, tempo):
    """
    Make a prediction for a new example
    
    TODO:
    - Create a sample input (you choose the values!)
    - Make a prediction
    - Print the input values and predicted output
    
    Args:
        model: trained model
        feature_names: list of feature names
    """
    print("\n" + "=" * 70)
    print("EXAMPLE PREDICTION")
    print("=" * 70)
    
    # Your code here
    # Example: If predicting house Popularity with [sqft, bedrooms, bathrooms]
    # sample = pd.DataFrame([[2000, 3, 2]], columns=feature_names)
    feature_names = pd.DataFrame([[duration, danceability, energy, tempo]], columns=['Duration', 'Danceability', 'Energy', 'Tempo'])
    predicted_popularity = model.predict(feature_names)[0]
    print(f"\n=== New Prediction ===")
    print(f"Song features: {duration:.0f} ms long, {danceability} danceability score, {energy} energy score, {tempo} tempo score")
    print(f"Predicted popularity: ${predicted_popularity:,.2f}")
    return predicted_popularity
    


if __name__ == "__main__":
    # Step 1: Load and explore
    data = load_and_explore_data(DATA_FILE)
    
    # Step 2: Visualize
    visualize_data(data)
    
    # Step 3: Prepare and split
    X_train, X_test, y_train, y_test = prepare_and_split_data(data)
    
    # Step 4: Train
    model = train_model(X_train, y_train)
    
    # Step 5: Evaluate
    predictions = evaluate_model(model, X_test, y_test)
    
    # Step 6: Make a prediction, add features as an argument
    make_prediction(model)
    
    print("\n" + "=" * 70)
    print("PROJECT COMPLETE!")
    print("=" * 70)
    print("\nNext steps:")
    print("1. Analyze your results")
    print("2. Try improving your model (add/remove features)")
    print("3. Create your presentation")
    print("4. Practice presenting with your group!")

