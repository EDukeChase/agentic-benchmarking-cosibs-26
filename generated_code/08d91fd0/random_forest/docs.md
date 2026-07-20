# Random Forest

Implementation: scikit-learn `RandomForestClassifier`.

Assumptions:
- Inputs are numeric or otherwise already encoded for tree-based learning.
- Missing values are not handled inside this module beyond what scikit-learn supports.

Limitations:
- Can be memory-heavy on large feature spaces.
- Feature importance is impurity-based only in this wrapper.
