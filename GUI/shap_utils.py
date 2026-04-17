import io
import zipfile
import numpy as np
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from ctrbp_model import EnhancedRNADataset, config
import base64


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

    feature_names = [f"Kmer:{k}" for k in kmer_list] + [f"Struct:{i}" for i in range(20)] + [f"PairProb:{i}" for i in range(20)]
    return X, feature_names, kmer_list


def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', pad_inches=0.02)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig)
    return img_base64


def generate_shap_images_base64(df, kmer_vocab, predictions):
    X, feature_names, kmer_list = prepare_features_from_df(df, kmer_vocab, max_samples=300)
    predictions = predictions[:X.shape[0]]

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

    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X_scaled, predictions)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_scaled)

    images = {}

    try:
        # 1. Bar Plot
        fig = plt.figure(figsize=(12, 6))
        shap.summary_plot(shap_values, X_scaled, feature_names=feature_names, plot_type="bar", show=False, max_display=20)
        plt.title("SHAP Feature Importance (Bar Plot)")
        images['bar'] = fig_to_base64(fig)
    except Exception as e:
        print(f"Bar plot error: {e}")
        images['bar'] = None

    try:
        # 2. Summary Plot (Dot)
        fig = plt.figure(figsize=(14, 8))
        shap.summary_plot(shap_values, X_scaled, feature_names=feature_names, show=False, max_display=15)
        plt.title("SHAP Summary Plot (Dot Plot)")
        images['summary'] = fig_to_base64(fig)
    except Exception as e:
        print(f"Summary plot error: {e}")
        images['summary'] = None

    try:
        # 3. Dependence Plot - Three subplots
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
        kmer_indices = list(range(n_kmer))
        struct_indices = list(range(n_kmer, n_kmer + 20))
        pairprob_indices = list(range(n_kmer + 20, n_kmer + 40))

        kmer_importance = mean_abs_shap[kmer_indices]
        top_kmer_idx = kmer_indices[int(np.argmax(kmer_importance))]
        top_kmer_name = feature_names[top_kmer_idx]

        struct_importance = mean_abs_shap[struct_indices]
        top_struct_idx = struct_indices[int(np.argmax(struct_importance))]
        top_struct_name = feature_names[top_struct_idx]

        pairprob_importance = mean_abs_shap[pairprob_indices]
        top_pairprob_idx = pairprob_indices[int(np.argmax(pairprob_importance))]
        top_pairprob_name = feature_names[top_pairprob_idx]

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        def plot_dependence(ax, feature_idx, shap_vals, X_data, feature_names, title):
            x = X_data[:, feature_idx]
            y = shap_vals[:, feature_idx]

            unique_vals = np.unique(x)
            is_binary = len(unique_vals) == 2 and set(unique_vals) == {0, 1}

            if is_binary:
                jitter = np.random.normal(0, 0.04, size=len(x))
                x_jittered = x + jitter
                sc = ax.scatter(x_jittered, y, c=y, cmap='coolwarm', alpha=0.7, s=25, edgecolors='w', linewidth=0.5)
                ax.set_xticks([0, 1])
                ax.set_xlim(-0.2, 1.2)
                mean_shap_0 = y[x == 0].mean() if np.any(x == 0) else 0
                mean_shap_1 = y[x == 1].mean() if np.any(x == 1) else 0
                ax.plot([0, 1], [mean_shap_0, mean_shap_1], 'k-', linewidth=2, label='Trend')
            else:
                sc = ax.scatter(x, y, c=y, cmap='coolwarm', alpha=0.7, s=25, edgecolors='w', linewidth=0.5)
                plt.colorbar(sc, ax=ax, label='SHAP value')
                z = np.polyfit(x, y, 2)
                p = np.poly1d(z)
                x_line = np.linspace(x.min(), x.max(), 100)
                ax.plot(x_line, p(x_line), 'k-', linewidth=2, label='Trend')

            ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.7)
            ax.set_xlabel(feature_names[feature_idx])
            ax.set_ylabel('SHAP Value')
            ax.set_title(title)
            ax.legend(loc='best')

        plot_dependence(
            axes[0], top_kmer_idx, shap_values, X_scaled, feature_names,
            title=f'K-mer: {top_kmer_name.replace("Kmer:", "")}'
        )
        plot_dependence(
            axes[1], top_struct_idx, shap_values, X_scaled, feature_names,
            title=f'Structure: {top_struct_name.replace("Struct:", "Pos ")}'
        )
        plot_dependence(
            axes[2], top_pairprob_idx, shap_values, X_scaled, feature_names,
            title=f'Pairing Prob: {top_pairprob_name.replace("PairProb:", "Pos ")}'
        )

        plt.tight_layout()
        images['dependence'] = fig_to_base64(fig)
    except Exception as e:
        print(f"Dependence plot error: {e}")
        images['dependence'] = None

    try:
        # 4. Force Plot - Highest probability
        highest_idx = int(np.argmax(predictions))
        base_value = explainer.expected_value

        shap.force_plot(
            base_value,
            shap_values[highest_idx],
            X_scaled[highest_idx],
            feature_names=feature_names,
            matplotlib=True,
            show=False
        )
        fig = plt.gcf()
        fig.set_size_inches(20, 4)
        plt.title(f"SHAP Force Plot - Highest Prediction (prob={predictions[highest_idx]:.3f})")
        plt.tight_layout()
        images['force_highest'] = fig_to_base64(fig)
        print(f"Force plot (highest) generated successfully for sample {highest_idx}")
    except Exception as e:
        print(f"Force plot (highest) error: {e}")
        import traceback
        traceback.print_exc()
        images['force_highest'] = None

    try:
        # 5. Force Plot - Lowest probability
        lowest_idx = int(np.argmin(predictions))
        base_value = explainer.expected_value

        shap.force_plot(
            base_value,
            shap_values[lowest_idx],
            X_scaled[lowest_idx],
            feature_names=feature_names,
            matplotlib=True,
            show=False
        )
        fig = plt.gcf()
        fig.set_size_inches(20, 4)
        plt.title(f"SHAP Force Plot - Lowest Prediction (prob={predictions[lowest_idx]:.3f})")
        plt.tight_layout()
        images['force_lowest'] = fig_to_base64(fig)
        print(f"Force plot (lowest) generated successfully for sample {lowest_idx}")
    except Exception as e:
        print(f"Force plot (lowest) error: {e}")
        import traceback
        traceback.print_exc()
        images['force_lowest'] = None

    try:
        # 6. Waterfall Plot - Highest probability
        highest_idx = int(np.argmax(predictions))
        base_value = explainer.expected_value
        explanation_high = shap.Explanation(
            values=shap_values[highest_idx],
            base_values=base_value,
            data=X_scaled[highest_idx],
            feature_names=feature_names
        )
        fig = plt.figure(figsize=(12, 8))
        shap.waterfall_plot(explanation_high, max_display=12, show=False)
        plt.title(f"SHAP Waterfall Plot - Highest Prediction (prob={predictions[highest_idx]:.3f})")
        images['waterfall_highest'] = fig_to_base64(fig)
    except Exception as e:
        print(f"Waterfall plot (highest) error: {e}")
        images['waterfall_highest'] = None

    try:
        # 7. Waterfall Plot - Lowest probability
        lowest_idx = int(np.argmin(predictions))
        base_value = explainer.expected_value
        explanation_low = shap.Explanation(
            values=shap_values[lowest_idx],
            base_values=base_value,
            data=X_scaled[lowest_idx],
            feature_names=feature_names
        )
        fig = plt.figure(figsize=(12, 8))
        shap.waterfall_plot(explanation_low, max_display=12, show=False)
        plt.title(f"SHAP Waterfall Plot - Lowest Prediction (prob={predictions[lowest_idx]:.3f})")
        images['waterfall_lowest'] = fig_to_base64(fig)
    except Exception as e:
        print(f"Waterfall plot (lowest) error: {e}")
        images['waterfall_lowest'] = None

    return images


def generate_shap_zip(df, kmer_vocab, predictions):
    X, feature_names, kmer_list = prepare_features_from_df(df, kmer_vocab, max_samples=300)
    predictions = predictions[:X.shape[0]]

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

    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X_scaled, predictions)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_scaled)

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        def add_plot(filename, plot_func, figsize=(12, 6)):
            fig = plt.figure(figsize=figsize)
            plot_func()
            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=300, bbox_inches='tight', pad_inches=0.02)
            buf.seek(0)
            zip_file.writestr(filename, buf.getvalue())
            plt.close(fig)

        add_plot('shap_bar.png',
                 lambda: shap.summary_plot(shap_values, X_scaled, feature_names=feature_names, plot_type="bar",
                                           show=False, max_display=20))

        add_plot('shap_summary.png',
                 lambda: shap.summary_plot(shap_values, X_scaled, feature_names=feature_names, show=False,
                                           max_display=15))

        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
        kmer_indices = list(range(n_kmer))
        struct_indices = list(range(n_kmer, n_kmer + 20))
        pairprob_indices = list(range(n_kmer + 20, n_kmer + 40))

        kmer_importance = mean_abs_shap[kmer_indices]
        top_kmer_idx = kmer_indices[int(np.argmax(kmer_importance))]
        top_kmer_name = feature_names[top_kmer_idx]

        struct_importance = mean_abs_shap[struct_indices]
        top_struct_idx = struct_indices[int(np.argmax(struct_importance))]
        top_struct_name = feature_names[top_struct_idx]

        pairprob_importance = mean_abs_shap[pairprob_indices]
        top_pairprob_idx = pairprob_indices[int(np.argmax(pairprob_importance))]
        top_pairprob_name = feature_names[top_pairprob_idx]

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        def plot_dependence_zip(ax, feature_idx, shap_vals, X_data, feature_names, title):
            x = X_data[:, feature_idx]
            y = shap_vals[:, feature_idx]

            unique_vals = np.unique(x)
            is_binary = len(unique_vals) == 2 and set(unique_vals) == {0, 1}

            if is_binary:
                jitter = np.random.normal(0, 0.04, size=len(x))
                x_jittered = x + jitter
                sc = ax.scatter(x_jittered, y, c=y, cmap='coolwarm', alpha=0.7, s=25, edgecolors='w', linewidth=0.5)
                ax.set_xticks([0, 1])
                ax.set_xlim(-0.2, 1.2)
                mean_shap_0 = y[x == 0].mean() if np.any(x == 0) else 0
                mean_shap_1 = y[x == 1].mean() if np.any(x == 1) else 0
                ax.plot([0, 1], [mean_shap_0, mean_shap_1], 'k-', linewidth=2, label='Trend')
            else:
                sc = ax.scatter(x, y, c=y, cmap='coolwarm', alpha=0.7, s=25, edgecolors='w', linewidth=0.5)
                plt.colorbar(sc, ax=ax, label='SHAP value')
                z = np.polyfit(x, y, 2)
                p = np.poly1d(z)
                x_line = np.linspace(x.min(), x.max(), 100)
                ax.plot(x_line, p(x_line), 'k-', linewidth=2, label='Trend')

            ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.7)
            ax.set_xlabel(feature_names[feature_idx])
            ax.set_ylabel('SHAP Value')
            ax.set_title(title)
            ax.legend(loc='best')

        plot_dependence_zip(
            axes[0], top_kmer_idx, shap_values, X_scaled, feature_names,
            title=f'K-mer: {top_kmer_name.replace("Kmer:", "")}'
        )
        plot_dependence_zip(
            axes[1], top_struct_idx, shap_values, X_scaled, feature_names,
            title=f'Structure: {top_struct_name.replace("Struct:", "Pos ")}'
        )
        plot_dependence_zip(
            axes[2], top_pairprob_idx, shap_values, X_scaled, feature_names,
            title=f'Pairing Prob: {top_pairprob_name.replace("PairProb:", "Pos ")}'
        )

        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight', pad_inches=0.02)
        buf.seek(0)
        zip_file.writestr('shap_dependence.png', buf.getvalue())
        plt.close(fig)

        highest_idx = int(np.argmax(predictions))
        lowest_idx = int(np.argmin(predictions))
        base_value = explainer.expected_value

        # Force plot highest
        shap.force_plot(base_value, shap_values[highest_idx], X_scaled[highest_idx],
                        feature_names=feature_names, matplotlib=True, show=False)
        fig = plt.gcf()
        fig.set_size_inches(20, 4)
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight', pad_inches=0.02)
        buf.seek(0)
        zip_file.writestr('shap_force_highest.png', buf.getvalue())
        plt.close(fig)

        # Force plot lowest
        shap.force_plot(base_value, shap_values[lowest_idx], X_scaled[lowest_idx],
                        feature_names=feature_names, matplotlib=True, show=False)
        fig = plt.gcf()
        fig.set_size_inches(20, 4)
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight', pad_inches=0.02)
        buf.seek(0)
        zip_file.writestr('shap_force_lowest.png', buf.getvalue())
        plt.close(fig)

        explanation_high = shap.Explanation(values=shap_values[highest_idx], base_values=base_value,
                                            data=X_scaled[highest_idx], feature_names=feature_names)
        add_plot('shap_waterfall_highest.png',
                 lambda: shap.waterfall_plot(explanation_high, max_display=12, show=False), figsize=(12, 8))

        explanation_low = shap.Explanation(values=shap_values[lowest_idx], base_values=base_value,
                                           data=X_scaled[lowest_idx], feature_names=feature_names)
        add_plot('shap_waterfall_lowest.png', lambda: shap.waterfall_plot(explanation_low, max_display=12, show=False),
                 figsize=(12, 8))

    zip_buffer.seek(0)
    return zip_buffer.getvalue()