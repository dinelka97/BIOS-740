from sklearn.metrics import classification_report

# Simulated predictions and true labels
y_true = ['AD', 'MCI', 'CN', 'AD', 'CN', 'MCI']
y_pred = ['AD', 'CN', 'CN', 'AD', 'MCI', 'MCI']

# Get classification report as a dictionary
report = classification_report(y_true, y_pred, output_dict=True)
print("Full report:", report)

# Extracting overall accuracy
accuracy = report['accuracy']
print("Overall accuracy:", accuracy)

# Extracting macro-average F1-score
f1_macro = report['macro avg']['f1-score']
print("Macro F1-score:", f1_macro)

# Extracting per-class F1-scores
f1_ad = report['AD']['f1-score']
f1_mci = report['MCI']['f1-score']
f1_cn = report['CN']['f1-score']

print("F1-score for AD:", f1_ad)
print("F1-score for MCI:", f1_mci)
print("F1-score for CN:", f1_cn)