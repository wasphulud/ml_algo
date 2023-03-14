import pandas as pd
from trees.decision_trees import DecisionTree, DecisionTreeParams
from ensemble.bagging import GenericBagging
import logging

LOGGING_LEVEL = logging.INFO
logging.basicConfig(
    level=LOGGING_LEVEL,
    format="%(asctime)s %(levelname)s %(module)s - %(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
data = pd.read_csv("data/bmi.csv")
data["Index"] = (data["Index"] >= 4) * 1
data["Index"] = data["Index"].astype("object")
training_set = data.sample(frac=0.6, random_state=42)
test_set = data.drop(training_set.index)
decision_tree = DecisionTree(
    decision_tree_params=DecisionTreeParams(max_depth=10), verbose=False
)
decision_tree.fit(training_set.drop(["Index"], axis=1), training_set["Index"])
predicted_values = decision_tree.predict(test_set)

gbagging = GenericBagging(
    model=DecisionTree(
        decision_tree_params=DecisionTreeParams(max_depth=6), verbose=False
    ),
    n_estimators=100,
    max_samples=0.6,
)
gbagging.fit(training_set.drop(["Index"], axis=1), training_set["Index"])


logging.warning(
    "Decision Tree %s",
    sum((test_set["Index"] * 1 == decision_tree.predict(test_set)) / len(test_set))
    * 100,
)


logging.warning(
    "bagging 1 %s",
    sum(
        (
            test_set["Index"].reset_index(drop=True)
            == gbagging.predict(test_set)["Predictions"].reset_index(drop=True)
        )
        / len(test_set)
    )
    * 100,
)
