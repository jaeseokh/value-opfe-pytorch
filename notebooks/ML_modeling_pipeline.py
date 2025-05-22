# ML_modeling.py
# One-stop pipeline to run 8 ML models for predicting yield-N response functions

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim

try:
    from pygam import LinearGAM, s
    pygam_available = True
except ImportError:
    pygam_available = False

# -----------------------------
# 1. Load and preprocess data
# -----------------------------

def load_data():
    project_root = Path("/Users/jaeseokhwang/value-ofpe-pytorch/value-opfe-pytorch")
    data_path = project_root / "Data" / "Processed" / "Analysis_ready"
    dat = pd.read_csv(data_path / "dat_binded.csv")

    dat = dat[dat["n_rate"] != 0].dropna()
    dat["nrate_gdd"] = dat["n_rate"] * dat["gdd_t"]
    dat["nrate_prcp"] = dat["n_rate"] * dat["prcp_t"]

    features = ["n_rate", "gdd_t", "prcp_t", "nrate_gdd", "nrate_prcp"]
    target = "yield"

    return dat, features, target

# -----------------------------
# 2. Preprocessing and splitting
# -----------------------------

def preprocess(train_df, test_df, features, target):
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_train = scaler_X.fit_transform(train_df[features].values)
    y_train = scaler_y.fit_transform(train_df[[target]].values)
    X_test = scaler_X.transform(test_df[features].values)
    y_test = scaler_y.transform(test_df[[target]].values)
    return X_train, y_train, X_test, y_test, scaler_X, scaler_y

def get_leave_one_out_splits(dat, group_col="ffy_id"):
    for g in dat[group_col].unique():
        yield g, dat[dat[group_col] != g], dat[dat[group_col] == g]

# -----------------------------
# 3. Models 
# -----------------------------


# -----------------------------
# 3a. Train MLP model
# -----------------------------

def train_mlp(X_train, y_train, X_test):
    # Step 1: Initialize the model with number of input features
    model = MLPModel(X_train.shape[1])

    # Step 2: Define loss function (Mean Squared Error)
    criterion = nn.MSELoss()

    # Step 3: Define optimizer (Adam optimizer for adaptive learning rate)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Step 4: Convert data from NumPy arrays to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

    # Step 5: Training loop (forward + backward pass)
    for epoch in range(200):
        model.train()                  # Set model to training mode
        optimizer.zero_grad()          # Clear gradients from previous step
        y_pred = model(X_train_tensor) # Forward pass
        loss = criterion(y_pred, y_train_tensor)  # Compute loss
        loss.backward()               # Backpropagation: compute gradients
        optimizer.step()              # Gradient descent step

    # Step 6: Prediction mode (no gradient tracking)
    model.eval()
    with torch.no_grad():
        preds = model(X_test_tensor).numpy()  # Forward pass only for test

    return model, preds

# --------------------------------------------------
# 📐 MLP Model: Key Equations for Reference (1 Hidden Layer)
# --------------------------------------------------
# Equation 1: z_j = ∑(w_ij^(1) * x_i) + b_j^(1)
# Equation 2: a_j = φ(z_j) = ReLU(z_j)
# Equation 3: ŷ = ∑(w_j^(2) * a_j) + b^(2)
# Full Model:
# ŷ = ∑(w_j^(2) * φ(∑(w_ij^(1) * x_i + b_j^(1))) + b^(2)
# Loss: L = (1/n) ∑(y_i - ŷ_i)^2

# -----------------------------
#  MLP architecture definition
# -----------------------------

class MLPModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        # Build a sequential model:
        # Input layer → Hidden layer (64) → ReLU → Hidden layer (32) → ReLU → Output layer (1)
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),  # Layer 1: input_dim → 64
            nn.ReLU(),                 # Activation after layer 1
            nn.Linear(64, 32),         # Layer 2: 64 → 32
            nn.ReLU(),                 # Activation after layer 2
            nn.Linear(32, 1)           # Output layer: 32 → 1 (predicted yield)
        )

    # Forward pass: defines how input flows through the model
    def forward(self, x):
        return self.model(x)


# -----------------------------
# 3b. Train TabNet model
# -----------------------------
# -----------------------------
# 📘 Math Basis for TabNet:
# -----------------------------
# TabNet combines gradient-based learning with feature selection using attention masks.
# Core idea: learn a sequence of decision steps where each step selects which input features to focus on.
# Each decision step:
#   - Applies a sparse mask (learned) to the feature vector
#   - Passes the masked features through nonlinear transformations (via decision blocks)
#   - Uses a loss + entropy term to encourage feature sparsity and interpretability
# Key Components:
#   - Shared layers: reuse across decision steps
#   - Step-dependent layers: unique to each step
#   - Attention: soft selection over features
#   - Objective: supervised loss + sparsity regularization
# -----------------------------

def train_tabnet(X_train, y_train, X_test):
    try:
        from pytorch_tabnet.tab_model import TabNetRegressor
    except ImportError:
        print(" TabNet not installed. Run: pip install pytorch-tabnet")
        return None, np.zeros_like(X_test[:, 0:1])

    model = TabNetRegressor(verbose=0)
    model.fit(X_train, y_train.ravel(), max_epochs=200)
    preds = model.predict(X_test).reshape(-1, 1)
    return model, preds


# -----------------------------
# 3c. Train Bayesian Neural Network model
# -----------------------------

# -----------------------------
# 📘 Math Basis for BNN:
# -----------------------------
# A BNN models uncertainty by treating the weights of a neural network as random variables (not fixed).
# Instead of learning point estimates for weights W, it learns a **posterior distribution** p(W | D)
# Key ideas:
#   - Prior: p(W) ~ N(0, I) often assumed
#   - Likelihood: p(y | x, W)
#   - Posterior: p(W | D) ∝ p(D | W) * p(W)
#   - Predictive distribution: p(y* | x*, D) = ∫ p(y* | x*, W) p(W | D) dW
# In practice:
#   - This integral is approximated using variational inference or sampling (e.g., MC Dropout, Pyro, TFP)
#   - Pyro + ELBO or Hamiltonian Monte Carlo are often used
# BNNs are powerful for quantifying epistemic uncertainty in small or high-stakes datasets
# -----------------------------

def train_bnn(X_train, y_train, X_test):
    # ⚠️ Placeholder: True BNNs require Pyro or TensorFlow Probability
    # This mock version returns zero predictions
    print(" BNN training placeholder — implement with Pyro or tfp later")
    return None, np.zeros_like(X_test[:, 0:1])


# -----------------------------
# 3d. Train Random Forest Model
# -----------------------------

# Equation: ŷ = Average(Tree₁(x), Tree₂(x), ..., Treeₖ(x))
# Key components:
#   - Ensemble of decision trees (bootstrap samples)
#   - Random feature selection at splits (reduces correlation)
#   - Reduces overfitting by averaging predictions
#   - Handles nonlinear interactions implicitly
# Implementation:
#   - sklearn.ensemble.RandomForestRegressor with n_estimators, max_depth

def train_rf(X_train, y_train, X_test):
    model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=0)
    model.fit(X_train, y_train.ravel())
    preds = model.predict(X_test).reshape(-1, 1)
    return model, preds

# -----------------------------
# 3e. Train Xgboost
# -----------------------------

# Equation: ŷ = ∑ₜ fₜ(x), where each fₜ is a regression tree
# Key components:
#   - Boosting: trees added sequentially to correct previous errors
#   - Gradient-based optimization (minimize loss)
#   - Regularization to prevent overfitting (L1/L2)
#   - Handles nonlinearities + feature interactions well
# Implementation:
#   - xgboost.XGBRegressor with n_estimators, learning_rate, max_depth

def train_xgb(X_train, y_train, X_test):
    try:
        from xgboost import XGBRegressor
    except ImportError:
        print(" XGBoost not installed. Run: pip install xgboost")
        return None, np.zeros_like(X_test[:, 0:1])

    model = XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, subsample=0.8)
    model.fit(X_train, y_train.ravel())
    preds = model.predict(X_test).reshape(-1, 1)
    return model, preds



# -----------------------------
# 4. Model dispatcher — maps string name to training function
# -----------------------------

def model_dispatcher():
    return {
        "mlp": train_mlp,
        "tabnet": train_tabnet,
        "bnn": train_bnn,
        "rf": train_rf,
        "xgb": train_xgb
    }


# -----------------------------
# 5. Main execution loop
# -----------------------------


# This loop runs the entire modeling pipeline:
#   - For each model (mlp, xgb, rf, etc.)
#   - For each unique field-year (leave-one-out)
#   - Train model on all other data, predict on the held-out field
#   - Print status or handle errors gracefully

if __name__ == "__main__":

    dat, features, target = load_data()  # Step 1: Load dataset and define features/target

    selected_ffy = dat['ffy_id'].unique()[:5]  # Step 2: Subset only 5 unique ffy_id for quick testing
    dat = dat[dat['ffy_id'].isin(selected_ffy)].copy()  # Filter dataset for selected fields

    dispatcher = model_dispatcher()  # Step 3: Get dictionary mapping model names to training functions
    results = []  # Step 4: Empty list to collect all predictions and metrics

    for model_name, train_fn in dispatcher.items():  # Step 5: Loop over models
        print(f"Running model: {model_name}")  # Log model name

        for ffy_id, train_df, test_df in get_leave_one_out_splits(dat):  # Step 6: Loop over field-year subsets (leave-one-out)
            try:
                # Step 7: Preprocess data for current split
                X_train, y_train, X_test, y_test, _, _ = preprocess(train_df, test_df, features, target)

                # Step 8: Train model and make predictions
                model, preds = train_fn(X_train, y_train, X_test)

                # Step 9: Evaluate RMSE for test predictions
                rmse = mean_squared_error(y_test, preds, squared=False)
                print(f"✅ Completed field {ffy_id} with model {model_name} | RMSE: {rmse:.4f}")

                # Step 10: Store results row by row
                for i in range(len(y_test)):
                    results.append({
                        "model": model_name,
                        "ffy_id": ffy_id,
                        "obs_id": i,
                        "true_yield_scaled": y_test[i][0],
                        "pred_yield_scaled": preds[i][0],
                        "rmse": rmse
                    })

            except Exception as e:
                print(f"❌ Error for field {ffy_id} with model {model_name}: {e}")  # Handle errors gracefully

    # Step 11: Save results to CSV file
    results_df = pd.DataFrame(results)
    results_df.to_csv("Results/model_predictions_sample.csv", index=False)
    print("📁 Results saved to Results/model_predictions_sample.csv")
