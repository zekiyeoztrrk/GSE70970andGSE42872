# Necessary libraries
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns

pandas2ri.activate()
geoquery = importr('GEOquery')

# Download datasets
gse70970 = geoquery.getGEO("GSE70970")
gse42872 = geoquery.getGEO("GSE42872")

# Extract expression data
gse70970_data = ro.r['exprs'](gse70970[0])
gse42872_data = ro.r['exprs'](gse42872[0])

# DataFrame
df1 = pandas2ri.rpy2py(gse70970_data)
df2 = pandas2ri.rpy2py(gse42872_data)

# Common genes
common_genes = df1.index.intersection(df2.index)
df1 = df1.loc[common_genes]
df2 = df2.loc[common_genes]

# Combining datasets
combined_df = pd.concat([df1, df2], axis=1)

# transposition (examples row, genes column)
combined_df = combined_df.T

# Create labels (0 for GSE70970, 1 for GSE42872)
labels = [0] * df1.shape[1] + [1] * df2.shape[1]

# splitting into training and test data
X_train, X_test, y_train, y_test = train_test_split(combined_df, labels, test_size=0.3, random_state=42)

# desicion tree
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
dt_preds = dt.predict(X_test)
print("Karar Ağacı Sınıflandırma Raporu:")
print(classification_report(y_test, dt_preds))

# svm
svc = SVC()
svc.fit(X_train, y_train)
svc_preds = svc.predict(X_test)
print("SVM Sınıflandırma Raporu:")
print(classification_report(y_test, svc_preds))

# random forest
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)
print("Rastgele Orman Sınıflandırma Raporu:")
print(classification_report(y_test, rf_preds))

# Feature importance levels
feature_importances = rf.feature_importances_
sorted_idx = feature_importances.argsort()
plt.figure(figsize=(10, 10))
plt.barh(combined_df.columns[sorted_idx], feature_importances[sorted_idx])
plt.xlabel("Rastgele Orman Özellik Önemi")
plt.show()

# Confusion matrix
def plot_confusion_matrix(y_true, y_pred, title):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Tahmin Edilen')
    plt.ylabel('Gerçek')
    plt.title(title)
    plt.show()

plot_confusion_matrix(y_test, dt_preds, "Desicion Tree Confusion Matrix")
plot_confusion_matrix(y_test, svc_preds, "SVM Confusion Matrix")
plot_confusion_matrix(y_test, rf_preds, "Random Forest Confusion Matrix")





