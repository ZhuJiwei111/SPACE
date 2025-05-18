import numpy as np
import os
import argparse
import glob
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import json
from tqdm import tqdm


def load_data(file_path):
    return np.load(file_path)


def train_random_forest(X, y, max_iter=1):
    clf = RandomForestClassifier(n_estimators=100, max_features="log2", random_state=42)
    skf = StratifiedKFold(n_splits=8, shuffle=True, random_state=42)
    scores = []

    for _ in range(max_iter):
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            clf.fit(X_train, y_train)
            y_pred = clf.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_pred)
            scores.append(auc)

    return scores


def main(directory, args):
    npy_files = glob.glob(os.path.join(directory, "*.npy"))
    npy_files.sort()

    tissue_aucs = []
    results_dict = {}

    for file_path in tqdm(npy_files):
        tissue_name = os.path.splitext(os.path.basename(file_path))[0]
        print(f"Processing {tissue_name}...")
        data = load_data(file_path)
        hidden_dim = data.shape[2]

        # Positive samples
        X_pos = data[:, 0, :]
        X_ref_pos = X_pos[:, : hidden_dim // 2]
        X_alt_pos = X_pos[:, hidden_dim // 2 :]
        y_pos = np.ones(X_pos.shape[0])
        # Negative samples
        X_neg = data[:, 1, :]
        X_ref_neg = X_neg[:, : hidden_dim // 2]
        X_alt_neg = X_neg[:, hidden_dim // 2 :]
        y_neg = np.zeros(X_neg.shape[0])

        # Combined samples
        X_combined = np.concatenate([X_pos, X_neg], axis=0)
        y_combined = np.concatenate([y_pos, y_neg], axis=0)

        # Ref samples
        X_ref_combined = np.concatenate([X_ref_pos, X_ref_neg], axis=0)

        # Alt samples
        X_alt_combined = np.concatenate([X_alt_pos, X_alt_neg], axis=0)

        # Minus samples
        X_minus_combined = X_alt_combined - X_ref_combined

        # Train and evaluate on combined data
        auc_combined = train_random_forest(X_combined, y_combined)
        mean_auc_combined = np.mean(auc_combined)

        # Train and evaluate on ref data
        auc_ref_combined = train_random_forest(X_ref_combined, y_combined)
        mean_auc_ref = np.mean(auc_ref_combined)

        # Train and evaluate on alt data
        auc_alt_combined = train_random_forest(X_alt_combined, y_combined)
        mean_auc_alt = np.mean(auc_alt_combined)

        # Train and evaluate on alt-ref data
        auc_minus_combined = train_random_forest(X_minus_combined, y_combined)
        mean_auc_minus = np.mean(auc_minus_combined)

        tissue_auc_mean = (
            mean_auc_combined + mean_auc_ref + mean_auc_alt + mean_auc_minus
        ) / 4
        tissue_aucs.append(tissue_auc_mean)

        print(f"Tissue {tissue_name} AUC (combined, ref, alt, minus): {mean_auc_combined}, {mean_auc_ref}, {mean_auc_alt}, {mean_auc_minus}")
        print(f"Tissue {tissue_name} Mean AUC: {tissue_auc_mean}")
        results_dict[tissue_name] = {
            "AUC_combined": mean_auc_combined,
            "AUC_ref": mean_auc_ref,
            "AUC_alt": mean_auc_alt,
            "AUC_minus": mean_auc_minus,
            "Mean_AUC": tissue_auc_mean,
        }

    print(f"Overall Mean AUC across all tissues: {np.mean(tissue_aucs)}")
    # 将字典保存到 JSON 文件中
    with open(f"{directory}.json", "w") as f:
        json.dump(results_dict, f, indent=4)
    # with open(f"{directory}.txt", "a+", encoding="utf-8") as f:
    #     f.write(f"{args.index} {mean_auc_minus}\n")


if __name__ == "__main__":
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(
        description="Calculate mean values from a JSON file."
    )
    parser.add_argument(
        "--directory", type=str, help="Path to the directory containing the JSON file."
    )
    parser.add_argument(
        "--index", type=str, help="Path to the directory containing the JSON file."
    )

    # 解析命令行参数
    args = parser.parse_args()

    # 调用主函数并传递目录参数
    main(args.directory, args)
