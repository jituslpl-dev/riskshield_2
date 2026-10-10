from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


class ClassifierFactory:
    """Factory: construct deterministic weighted classifiers behind one contract."""

    @staticmethod
    def create(ratio: float):
        return {
            "logistic": make_pipeline(
                StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
            ),
            "random_forest": RandomForestClassifier(
                n_estimators=120, max_depth=8, class_weight="balanced", random_state=42, n_jobs=2
            ),
            "xgboost": XGBClassifier(
                n_estimators=120,
                max_depth=4,
                learning_rate=0.05,
                scale_pos_weight=ratio,
                random_state=42,
                n_jobs=2,
            ),
            "lightgbm": LGBMClassifier(
                n_estimators=120,
                max_depth=4,
                num_leaves=15,
                learning_rate=0.05,
                scale_pos_weight=ratio,
                random_state=42,
                n_jobs=2,
                verbosity=-1,
            ),
            "catboost": CatBoostClassifier(
                iterations=120,
                depth=4,
                learning_rate=0.05,
                class_weights=[1, float(ratio)],
                random_seed=42,
                verbose=False,
                thread_count=2,
                allow_writing_files=False,
            ),
        }
