"""
Statistical analysis of random search results with multiple regression.
Analyzes relationships between connection parameters and firing rates/CV of ISIs.
"""

import glob
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import seaborn as sns

def find_latest_random_search_results():
    """Find the most recent random_search_summary.pkl file"""
    base_folder = os.path.join(os.path.dirname(__file__), "Runs", "random_search")
    search_pattern = os.path.join(base_folder, "random_search_EtoI_ItoE_E*_I*", "*", "random_search_summary.pkl")
    summary_files = glob.glob(search_pattern)
    
    if not summary_files:
        raise FileNotFoundError(f"No random_search_summary.pkl files found under {base_folder}")
    
    latest_file = max(summary_files, key=os.path.getmtime)
    latest_path = os.path.dirname(latest_file)
    return latest_file, latest_path

def extract_features_and_targets(successful_results):
    """Extract features (connection parameters) and targets (metrics) from results"""
    
    # Extract quadrant means (these are the actual connection strengths)
    ee_means = np.array([r.get('quadrant_means', {}).get('EE', np.nan) for r in successful_results])
    ei_means = np.array([r.get('quadrant_means', {}).get('EI', np.nan) for r in successful_results])
    ie_means = np.array([r.get('quadrant_means', {}).get('IE', np.nan) for r in successful_results])
    ii_means = np.array([r.get('quadrant_means', {}).get('II', np.nan) for r in successful_results])
    
    # Take absolute values for inhibitory connections (they're negative)
    ee_means = np.abs(ee_means)
    ei_means = np.abs(ei_means)
    ie_means = np.abs(ie_means)
    ii_means = np.abs(ii_means)
    
    # Extract targets (metrics to predict)
    e_rates = np.array([r.get('e_rate', np.nan) for r in successful_results])
    i_rates = np.array([r.get('i_rate', np.nan) for r in successful_results])
    cv_e = np.array([r.get('cv_e', np.nan) for r in successful_results])
    cv_i = np.array([r.get('cv_i', np.nan) for r in successful_results])
    
    # Build feature matrix: 4 connection means + 6 ratios
    n_samples = len(successful_results)
    features = np.zeros((n_samples, 10))
    feature_names = [
        "EE", "EI", "IE", "II",
        "EE/EI", "EE/IE", "EE/II",
        "EI/IE", "EI/II", "IE/II"
    ]
    
    features[:, 0] = ee_means
    features[:, 1] = ei_means
    features[:, 2] = ie_means
    features[:, 3] = ii_means
    features[:, 4] = np.where(ei_means > 1e-6, ee_means / ei_means, np.nan)  # EE/EI
    features[:, 5] = np.where(ie_means > 1e-6, ee_means / ie_means, np.nan)  # EE/IE
    features[:, 6] = np.where(ii_means > 1e-6, ee_means / ii_means, np.nan)  # EE/II
    features[:, 7] = np.where(ie_means > 1e-6, ei_means / ie_means, np.nan)  # EI/IE
    features[:, 8] = np.where(ii_means > 1e-6, ei_means / ii_means, np.nan)  # EI/II
    features[:, 9] = np.where(ii_means > 1e-6, ie_means / ii_means, np.nan)  # IE/II
    
    # Create targets dictionary
    targets = {
        'E_rate': e_rates,
        'I_rate': i_rates,
        'CV_e': cv_e,
        'CV_i': cv_i
    }
    
    return features, feature_names, targets

def print_correlation_summary(features, feature_names, targets, save_folder):
    """Print and save correlation summary between features and targets"""
    
    print("\n" + "="*80)
    print("CORRELATION SUMMARY")
    print("="*80)
    
    # Create log-transformed features for correlation
    log_features = np.log10(np.abs(features) + 1e-6)
    
    results_text = []
    results_text.append("\nPearson Correlations:\n")
    
    for target_name, target_values in targets.items():
        if np.all(np.isnan(target_values)):
            continue
            
        results_text.append(f"\n{target_name}:")
        print(f"\n{target_name}:")
        
        for i, feat_name in enumerate(feature_names[:4]):  # Just show the 4 main connections
            # Remove NaN values
            valid_mask = np.isfinite(log_features[:, i]) & np.isfinite(target_values)
            if np.sum(valid_mask) < 3:
                continue
                
            corr, pval = pearsonr(log_features[valid_mask, i], target_values[valid_mask])
            if not np.isnan(corr):
                results_text.append(f"  vs {feat_name:8}: r={corr:7.3f}, p={pval:.3e}")
                print(f"  vs {feat_name:8}: r={corr:7.3f}, p={pval:.3e}")
    
    # Save to file
    results_file = os.path.join(save_folder, "correlation_summary.txt")
    with open(results_file, 'w') as f:
        f.write("".join(results_text))
    
    print(f"\nCorrelation summary saved to: {results_file}")

def multiple_regression_analysis(features, feature_names, targets, save_folder):
    """Perform multiple regression analysis for each target metric"""
    
    print("\n" + "="*80)
    print("MULTIPLE REGRESSION ANALYSIS")
    print("="*80)
    
    # Log-transform features for regression
    log_features = np.log10(np.abs(features) + 1e-6)
    
    # Create figure with subplots for each target
    n_targets = len([t for t in targets.values() if not np.all(np.isnan(t))])
    fig, axes = plt.subplots(n_targets, 1, figsize=(14, 5 * n_targets), squeeze=False)
    fig.suptitle('Multiple Regression: Connection Parameters vs Metrics', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    target_idx = 0
    regression_results = {}
    
    for target_name, target_values in targets.items():
        if np.all(np.isnan(target_values)):
            continue
        
        # Filter out NaN values
        valid_mask = np.isfinite(target_values)
        for j in range(log_features.shape[1]):
            valid_mask = valid_mask & np.isfinite(log_features[:, j])
        
        if np.sum(valid_mask) < len(feature_names) + 2:  # Need at least n_features + 2 samples
            print(f"\nSkipping {target_name}: insufficient valid data")
            continue
        
        X_valid = log_features[valid_mask, :]
        y_valid = target_values[valid_mask]
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_valid)
        
        # Fit multiple regression
        reg = LinearRegression()
        reg.fit(X_scaled, y_valid)
        
        # Calculate R²
        y_pred = reg.predict(X_scaled)
        ss_res = np.sum((y_valid - y_pred) ** 2)
        ss_tot = np.sum((y_valid - np.mean(y_valid)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Store results
        regression_results[target_name] = {
            'coefficients': reg.coef_,
            'intercept': reg.intercept_,
            'r_squared': r_squared,
            'feature_names': feature_names
        }
        
        # Print results
        print(f"\n{target_name}:")
        print(f"  R² = {r_squared:.4f}")
        print(f"  Intercept = {reg.intercept_:.4f}")
        print(f"  Coefficients:")
        for i, (feat_name, coef) in enumerate(zip(feature_names, reg.coef_)):
            print(f"    {feat_name:8}: {coef:8.4f}")
        
        # Plot coefficients
        ax = axes[target_idx, 0]
        coef_values = reg.coef_
        colors = ["red" if c < 0 else "blue" for c in coef_values]
        bars = ax.barh(range(len(feature_names)), coef_values, color=colors, alpha=0.7)
        ax.set_yticks(range(len(feature_names)))
        ax.set_yticklabels(feature_names, fontsize=10)
        ax.set_xlabel("Standardized Coefficient", fontsize=12, fontweight='bold')
        ax.set_title(f"{target_name} (R² = {r_squared:.4f})", fontsize=13, fontweight='bold')
        ax.axvline(x=0, color="black", linestyle="-", linewidth=1)
        ax.grid(True, axis="x", linestyle="--", alpha=0.3)
        
        # Add coefficient values as text
        for i, (bar, val) in enumerate(zip(bars, coef_values)):
            ax.text(val, i, f" {val:.3f}", va="center", 
                   ha="left" if val >= 0 else "right", fontsize=9)
        
        target_idx += 1
    
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    
    save_path = os.path.join(save_folder, "multiple_regression_analysis.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nMultiple regression plot saved to: {save_path}")
    plt.close()
    
    # Save regression results to file
    results_file = os.path.join(save_folder, "regression_results.txt")
    with open(results_file, 'w') as f:
        f.write("MULTIPLE REGRESSION RESULTS\n")
        f.write("="*80 + "\n\n")
        
        for target_name, results in regression_results.items():
            f.write(f"{target_name}:\n")
            f.write(f"  R² = {results['r_squared']:.4f}\n")
            f.write(f"  Intercept = {results['intercept']:.4f}\n")
            f.write(f"  Coefficients:\n")
            for feat_name, coef in zip(results['feature_names'], results['coefficients']):
                f.write(f"    {feat_name:8}: {coef:8.4f}\n")
            f.write("\n")
    
    print(f"Regression results saved to: {results_file}")
    
    return regression_results

def create_scatter_plots(features, feature_names, targets, save_folder):
    """Create comprehensive scatter plots including ratios and sums of incoming connections"""
    
    print("\n" + "="*80)
    print("CREATING SCATTER PLOTS")
    print("="*80)
    
    # Extract base connections (EE, EI, IE, II)
    ee = np.abs(features[:, 0])
    ei = np.abs(features[:, 1])
    ie = np.abs(features[:, 2])
    ii = np.abs(features[:, 3])
    
    # Compute sums of incoming connections
    incoming_to_E = ee + ie  # EE + IE
    incoming_to_I = ei + ii  # EI + II
    incoming_ratio = np.where(incoming_to_I > 1e-6, incoming_to_E / incoming_to_I, np.nan)  # (EE+IE)/(EI+II)
    
    # Build comprehensive feature list
    all_features = {
        # Main connections
        'EE': ee,
        'EI': ei,
        'IE': ie,
        'II': ii,
        # Ratios
        'EE/EI': features[:, 4],
        'EE/IE': features[:, 5],
        'EE/II': features[:, 6],
        'EI/IE': features[:, 7],
        'EI/II': features[:, 8],
        'IE/II': features[:, 9],
        # Sums of incoming
        'EE+IE': incoming_to_E,
        'EI+II': incoming_to_I,
        '(EE+IE)/(EI+II)': incoming_ratio
    }
    
    all_feature_names = list(all_features.keys())
    n_features = len(all_feature_names)
    n_targets = len([t for t in targets.values() if not np.all(np.isnan(t))])
    
    # Create comprehensive scatter plot grid
    fig, axes = plt.subplots(n_targets, n_features, figsize=(3 * n_features, 3 * n_targets))
    if n_targets == 1:
        axes = axes.reshape(1, -1)
    elif n_features == 1:
        axes = axes.reshape(-1, 1)
    
    fig.suptitle('Comprehensive Scatter Plots: Connection Parameters, Ratios, and Sums vs Metrics', 
                 fontsize=14, fontweight='bold', y=0.995)
    
    target_idx = 0
    for target_name, target_values in targets.items():
        if np.all(np.isnan(target_values)):
            continue
        
        for feat_idx, feat_name in enumerate(all_feature_names):
            ax = axes[target_idx, feat_idx]
            
            x_data_raw = all_features[feat_name]
            
            # Use linear scale (no log transform)
            x_data = np.abs(x_data_raw)
            
            # Filter out NaN values
            valid_mask = np.isfinite(x_data) & np.isfinite(target_values) & np.isfinite(x_data_raw)
            
            if np.sum(valid_mask) < 3:
                ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center", 
                       transform=ax.transAxes, fontsize=8)
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            
            x_valid = x_data[valid_mask]
            y_valid = target_values[valid_mask]
            
            ax.scatter(x_valid, y_valid, alpha=0.6, s=20, edgecolors='none')
            ax.set_xlabel(f'{feat_name}', fontsize=8)
            ax.set_ylabel(target_name, fontsize=8)
            ax.grid(True, alpha=0.3, linewidth=0.5)
            ax.tick_params(labelsize=7)
            
            # Add correlation
            corr, pval = pearsonr(x_valid, y_valid)
            if not np.isnan(corr):
                # Truncate p-value display for very small values
                if pval < 0.001:
                    p_str = f'p<0.001'
                else:
                    p_str = f'p={pval:.3f}'
                ax.set_title(f'r={corr:.3f}\n{p_str}', fontsize=7, pad=2)
        
        target_idx += 1
    
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    
    save_path = os.path.join(save_folder, "scatter_plots.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Comprehensive scatter plots saved to: {save_path}")
    plt.close()

if __name__ == "__main__":
    print("="*80)
    print("RANDOM SEARCH STATISTICAL ANALYSIS WITH MULTIPLE REGRESSION")
    print("="*80)
    
    # Find and load the latest results
    try:
        results_file, results_folder = find_latest_random_search_results()
        print(f"\nLoading results from: {results_file}")
        
        with open(results_file, 'rb') as f:
            all_results = pickle.load(f)
        
        print(f"Loaded {len(all_results)} results")
        successful_results = [r for r in all_results if r.get('success', False)]
        print(f"Successful runs: {len(successful_results)}/{len(all_results)}")
        
        if len(successful_results) < 10:
            print(f"\nWARNING: Only {len(successful_results)} successful runs. Multiple regression needs more data.")
        
        # Create analysis folder
        analysis_folder = os.path.join(results_folder, "Analysis")
        os.makedirs(analysis_folder, exist_ok=True)
        
        # Extract features and targets
        features, feature_names, targets = extract_features_and_targets(successful_results)
        
        # Run analyses
        print_correlation_summary(features, feature_names, targets, analysis_folder)
        regression_results = multiple_regression_analysis(features, feature_names, targets, analysis_folder)
        create_scatter_plots(features, feature_names, targets, analysis_folder)
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE")
        print("="*80)
        print(f"Results saved to: {analysis_folder}")
        print("="*80 + "\n")
        
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("\nPlease run recurrent/parameter_search/random_search.py first to generate simulation data.")

