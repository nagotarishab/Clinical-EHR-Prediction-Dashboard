import copy
import logging
import os

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample, shuffle

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class ModelPipeline:
    """
    Decision Tree, linear SVM (SGD hinge), and MLP — trained on Dataset 1 train,
    evaluated with standard metrics, ROC-ready scores, and continual updates on Dataset 2 train.
    """

    def __init__(self, random_state=42, artifact_dir=None, dt_max_depth=10):
        self.random_state = random_state
        self.dt_max_depth = dt_max_depth
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.artifact_dir = artifact_dir or os.path.join(base, "artifacts")
        os.makedirs(self.artifact_dir, exist_ok=True)

        self.dt_model = DecisionTreeClassifier(
            max_depth=dt_max_depth,
            class_weight="balanced",
            min_samples_leaf=5,
            random_state=self.random_state,
        )
        self.svm_model = SGDClassifier(
            loss="hinge",
            penalty="l2",
            alpha=0.001,
            max_iter=2000,
            class_weight="balanced",
            random_state=self.random_state,
        )
        self.mlp_model = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            alpha=0.001,
            batch_size=256,
            max_iter=2000,
            early_stopping=False,
            warm_start=True,
            random_state=self.random_state,
        )

        self.models = {
            "Decision Tree": self.dt_model,
            "SVM (SGD)": self.svm_model,
            "MLP": self.mlp_model,
        }
        self._score_thresholds = {}

    def _balance_data(self, X, y):
        """Manually oversample the minority class for MLP since it doesn't support sample_weight."""
        classes, counts = np.unique(y, return_counts=True)
        if len(classes) < 2:
            return X, y
            
        majority_class = classes[np.argmax(counts)]
        minority_class = classes[np.argmin(counts)]
        
        X_majority = X[y == majority_class]
        y_majority = y[y == majority_class]
        X_minority = X[y == minority_class]
        y_minority = y[y == minority_class]
        
        X_minority_upsampled, y_minority_upsampled = resample(
            X_minority, y_minority, 
            replace=True, 
            n_samples=len(y_majority), 
            random_state=self.random_state
        )
        
        X_balanced = np.vstack((X_majority, X_minority_upsampled))
        y_balanced = np.hstack((y_majority, y_minority_upsampled))
        X_balanced, y_balanced = shuffle(X_balanced, y_balanced, random_state=self.random_state)
        
        return X_balanced, y_balanced

    def _tune_threshold(self, model, X_tune, y_tune):
        """Pick a score threshold from the train-set PR curve (F1-optimal; rare positives)."""
        scores = self._scores_for_roc(model, X_tune)
        if scores is None or len(np.unique(y_tune)) < 2:
            return None
        if np.sum(y_tune == 1) == 0 or np.sum(y_tune == 0) == 0:
            return float(np.median(scores))
        prec, rec, thr = precision_recall_curve(y_tune, scores)
        if len(thr) == 0:
            return float(np.median(scores))
        # len(prec)==len(rec)==len(thr)+1 — pair first len(thr) points with thr
        f1s = (2.0 * prec[:-1] * rec[:-1]) / (prec[:-1] + rec[:-1] + 1e-12)
        j = int(np.nanargmax(f1s))
        return float(thr[j])

    def retune_thresholds(self, X_tune, y_tune):
        """Re-fit F1-optimal score thresholds after continual learning."""
        for name, model in self.models.items():
            t = self._tune_threshold(model, X_tune, y_tune)
            if t is not None:
                self._score_thresholds[name] = t

    def _scores_for_roc(self, model, X):
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)
            if proba.shape[1] > 1:
                return proba[:, 1]
            return proba[:, 0]
        if hasattr(model, "decision_function"):
            return model.decision_function(X)
        return None

    def evaluate(self, y_true, y_pred, y_score=None):
        metrics = {
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred, zero_division=0),
            "Recall": recall_score(y_true, y_pred, zero_division=0),
            "F1-Score": f1_score(y_true, y_pred, zero_division=0),
        }
        if y_score is not None and len(np.unique(y_true)) > 1:
            try:
                metrics["ROC-AUC"] = roc_auc_score(y_true, y_score)
            except ValueError:
                metrics["ROC-AUC"] = float("nan")
        else:
            metrics["ROC-AUC"] = float("nan")
        return metrics

    def evaluate_models(self, X, y, dataset_name=""):
        logging.info("Evaluating models on %s", dataset_name)
        out = {}
        for name, model in self.models.items():
            y_score = self._scores_for_roc(model, X)
            thr = self._score_thresholds.get(name)
            if y_score is not None and thr is not None:
                y_pred = (y_score >= thr).astype(np.int32)
            else:
                y_pred = model.predict(X)
            out[name] = {
                "metrics": self.evaluate(y, y_pred, y_score),
                "y_pred": y_pred,
                "y_score": y_score,
                "confusion": confusion_matrix(y, y_pred, labels=[0, 1]),
            }
        return out

    def train_base_models(self, X_train, y_train):
        logging.info("Training base models on Dataset 1 train (class-weighted; real label counts)...")
        classes = np.unique(y_train)
        if len(classes) < 2:
            raise ValueError(
                "Dataset 1 train contains only one class after the temporal split. "
                "Pick a more frequent target (e.g. Sinusitis) or adjust the split date."
            )

        for name, model in self.models.items():
            logging.info("Training %s...", name)
            if name == "MLP":
                X_bal, y_bal = self._balance_data(X_train, y_train)
                model.fit(X_bal, y_bal)
                thr = 0.5  # Since it's trained on balanced data, 0.5 is the natural boundary
            else:
                model.fit(X_train, y_train)
                thr = self._tune_threshold(model, X_train, y_train)
                
            self._score_thresholds[name] = thr
            logging.info("%s done (score threshold=%s).", name, thr)

        self.save_models(X_train.shape[1])

    def save_models(self, n_features=None):
        bundle = {name: copy.deepcopy(m) for name, m in self.models.items()}
        config = {
            "n_features": n_features,
            "dt_max_depth": self.dt_max_depth,
            "score_thresholds": {k: float(v) for k, v in self._score_thresholds.items() if v is not None},
            "decision_tree": {
                "class_weight": "balanced",
                "min_samples_leaf": 5,
                "max_depth": self.dt_max_depth,
            },
            "sgd_svm": {
                "loss": "hinge",
                "penalty": "l2",
                "alpha": 0.001,
                "max_iter": 2000,
                "class_weight": "balanced",
            },
            "mlp": {
                "hidden_layer_sizes": (64, 32),
                "activation": "relu",
                "solver": "adam",
                "alpha": 0.001,
                "batch_size": 256,
                "max_iter": 2000,
                "warm_start": True,
            },
        }
        path = os.path.join(self.artifact_dir, "dataset1_models.joblib")
        joblib.dump({"models": bundle, "config": config}, path)
        logging.info("Saved trained models + config to %s", path)

    def load_models(self, path=None):
        path = path or os.path.join(self.artifact_dir, "dataset1_models.joblib")
        if not os.path.isfile(path):
            return False
        raw = joblib.load(path)
        if isinstance(raw, dict) and "models" in raw:
            bundle = raw["models"]
            cfg = raw.get("config") or {}
            self._score_thresholds = cfg.get("score_thresholds", {}) or {}
        else:
            bundle = raw
        for name, m in bundle.items():
            if name in self.models:
                self.models[name] = m
        self.dt_model = self.models["Decision Tree"]
        self.svm_model = self.models["SVM (SGD)"]
        self.mlp_model = self.models["MLP"]
        return True

    def continual_learning_update(self, X_d2_train, y_d2_train, X_d1_train, y_d1_train):
        """
        Fine-tune from Dataset 1 checkpoints: tree refit on D1-train ∪ D2-train; SGD/MLP partial_fit on D2 train;
        score thresholds re-tuned on combined train. No held-out test data used here.
        """
        logging.info("Continual learning on Dataset 2 train...")
        y_comb = np.hstack((y_d1_train, y_d2_train))
        classes = np.unique(y_comb)

        can_partial = len(np.unique(y_d2_train)) >= 2

        for name, model in self.models.items():
            if name == "Decision Tree":
                X_comb = np.vstack((X_d1_train, X_d2_train))
                y_comb_dt = np.hstack((y_d1_train, y_d2_train))
                model.fit(X_comb, y_comb_dt)
            elif can_partial:
                if name == "MLP":
                    X_d2_bal, y_d2_bal = self._balance_data(X_d2_train, y_d2_train)
                    model.partial_fit(X_d2_bal, y_d2_bal, classes=classes)
                else:
                    model.partial_fit(X_d2_train, y_d2_train, classes=classes)
            else:
                logging.warning(
                    "%s: Dataset 2 train is single-class; refitting on D1+D2 train instead of partial_fit.",
                    name,
                )
                X_comb = np.vstack((X_d1_train, X_d2_train))
                y_comb_all = np.hstack((y_d1_train, y_d2_train))
                if len(np.unique(y_comb_all)) >= 2:
                    if name == "MLP":
                        X_comb_bal, y_comb_bal = self._balance_data(X_comb, y_comb_all)
                        model.fit(X_comb_bal, y_comb_bal)
                    else:
                        model.fit(X_comb, y_comb_all)

        logging.info("Continual learning complete.")
        Xtune = np.vstack((X_d1_train, X_d2_train))
        ytune = np.hstack((y_d1_train, y_d2_train))
        self.retune_thresholds(Xtune, ytune)

        joblib.dump(
            {
                "models": {n: copy.deepcopy(m) for n, m in self.models.items()},
                "config": {
                    "score_thresholds": {k: float(v) for k, v in self._score_thresholds.items() if v is not None},
                    "stage": "continual",
                },
            },
            os.path.join(self.artifact_dir, "continual_models.joblib"),
        )

    def decision_tree_importance(self, feature_names):
        imp = self.dt_model.feature_importances_
        order = np.argsort(imp)[::-1][:25]
        return [(feature_names[i] if i < len(feature_names) else str(i), float(imp[i])) for i in order]

    def svm_linear_weights(self, feature_names, top_k=18):
        """Absolute linear coefficients (binary); proxy for which directions the margin uses."""
        coef = getattr(self.svm_model, "coef_", None)
        if coef is None:
            return []
        w = np.abs(np.ravel(coef))
        k = min(top_k, len(w))
        order = np.argsort(w)[::-1][:k]
        return [(feature_names[i] if i < len(feature_names) else str(i), float(w[i])) for i in order]

    def mlp_permutation_importance(self, X_test, y_test, feature_names, n_repeats=5):
        from sklearn.inspection import permutation_importance
        perm_imp = permutation_importance(
            self.mlp_model, X_test, y_test,
            n_repeats=n_repeats, random_state=self.random_state, scoring='accuracy'
        )
        w = perm_imp.importances_mean
        order = np.argsort(w)[::-1][:20]
        return [(feature_names[i] if i < len(feature_names) else str(i), float(w[i])) for i in order]


    @staticmethod
    def roc_curve_data(y_true, y_score):
        if y_score is None or len(np.unique(y_true)) < 2:
            return None, None, None
        fpr, tpr, thr = roc_curve(y_true, y_score)
        auc = roc_auc_score(y_true, y_score)
        return fpr, tpr, auc
