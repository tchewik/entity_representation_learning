import pandas as pd
from sklearn.metrics import classification_report


def report_pd(true, pred):
    report = classification_report(true, pred, digits=3, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    return report_df
