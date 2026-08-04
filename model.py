class LinearWeightModel:
    """Simple linear regression model: weight = coef * height + intercept."""

    def __init__(self, coef: float, intercept: float):
        self.coef = float(coef)
        self.intercept = float(intercept)

    def predict(self, X):
        # X shape: [[height], ...]
        results = []
        for row in X:
            height = float(row[0])
            results.append([self.coef * height + self.intercept])
        return results
