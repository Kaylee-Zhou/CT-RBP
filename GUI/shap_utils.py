import io
import zipfile
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from ctrbp_model import EnhancedRNADataset, config


def prepare_features_from_df(df, kmer_vocab, max_samples=300):
    dataset = EnhancedRNADataset(df, kmer_vocab=kmer_vocab, max_length=config.max_length)
    n_samples = min(max_samples, len(dataset))
    reverse_vocab = {v: k for k, v in kmer_vocab.items()}

    all_kmers = set()
    for i in range(n_samples):
        sample = dataset[i]
        seq_indices = sample['sequence'].numpy()
        for idx in seq_indices:
            if idx in reverse_vocab:
                kmer = reverse_vocab[idx]
                if kmer not in ['<PAD>', '<UNK>'] and len(kmer) == 3:
                    all_kmers.add(kmer)
    kmer_list = sorted(list(all_kmers))

    X = []
    for i in range(n_samples):
        sample = dataset[i]
        seq_indices = sample['sequence'].numpy()
        kmer_presence = np.zeros(len(kmer_list))
        for idx in seq_indices:
            if idx in reverse_vocab:
                kmer = reverse_vocab[idx]
                if kmer in kmer_list:
                    kmer_presence[kmer_list.index(kmer)] = 1
        structure = sample['structure'].numpy()[:20]
        pairing_prob = sample['pairing_prob'].numpy()[:20]
        feature_vec = list(kmer_presence) + list(structure) + list(pairing_prob)
        X.append(feature_vec)
    X = np.array(X)

    feature_names = [f"Kmer:{k}" for k in kmer_list] + [f"Struct:{i}" for i in range(20)] + [f"PairProb:{i}" for i in
                                                                                             range(20)]
    return X, feature_names, kmer_list

def generate_shap_zip(df, kmer_vocab, predictions):
    X, feature_names, kmer_list = prepare_features_from_df(df, kmer_vocab, max_samples=300)
    predictions = predictions[:X.shape[0]]

    # Standardized continuous features
    n_kmer = len(kmer_list)
    n_structure = 20
    kmer_feat = X[:, :n_kmer]
    struct_feat = X[:, n_kmer:n_kmer + n_structure]
    pair_feat = X[:, n_kmer + n_structure:]
    scaler_struct = StandardScaler()
    struct_scaled = scaler_struct.fit_transform(struct_feat)
    scaler_pair = StandardScaler()
    pair_scaled = scaler_pair.fit_transform(pair_feat)
    X_scaled = np.hstack([kmer_feat, struct_scaled, pair_scaled])

    # Train the model
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X_scaled, predictions)

    # SHAP Interpreter
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_scaled)

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # Auxiliary function
        def add_plot(filename, plot_func, figsize=(12, 6)):
            plt.figure(figsize=figsize)
            plot_func()
            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
            buf.seek(0)
            zip_file.writestr(filename, buf.getvalue())
            plt.close()

        # 1. Bar Plot
        add_plot('shap_bar.png',
                 lambda: shap.summary_plot(shap_values, X_scaled, feature_names=feature_names, plot_type="bar",
                                           show=False, max_display=20))

        # 2. Summary Plot (Dot)
        add_plot('shap_summary.png',
                 lambda: shap.summary_plot(shap_values, X_scaled, feature_names=feature_names, show=False,
                                           max_display=20))

        # 3. Dependence Plot (The most important feature)
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
        top_idx = np.argmax(mean_abs_shap)
        add_plot('shap_dependence.png',
                 lambda: shap.dependence_plot(top_idx, shap_values, X_scaled, feature_names=feature_names, show=False),
                 figsize=(10, 6))

        # 4. Force Plot (The sample with the highest prediction probability)
        idx = np.argmax(predictions)
        base_value = explainer.expected_value
        plt.figure(figsize=(20, 4))
        shap.force_plot(base_value, shap_values[idx], X_scaled[idx], feature_names=feature_names, matplotlib=True,
                        show=False)
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        zip_file.writestr('shap_force.png', buf.getvalue())
        plt.close()

        # 5. Waterfall Plot
        explanation = shap.Explanation(values=shap_values[idx], base_values=base_value, data=X_scaled[idx],
                                       feature_names=feature_names)
        add_plot('shap_waterfall.png', lambda: shap.waterfall_plot(explanation, max_display=12, show=False),
                 figsize=(12, 8))

    zip_buffer.seek(0)
    return zip_buffer.getvalue()