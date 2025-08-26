import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score,recall_score,f1_score,confusion_matrix
df=pd.read_csv("crop.csv")
print(df.head(3))

print(df["district"].nunique())

# sns.pairplot(data=df,hue="label")
# plt.show()
# sns.boxplot(data=df)
# plt.show()

x=df[["N","P","K","temperature","humidity","ph","rainfall"]]

y=df["label"]
le=LabelEncoder()
y=le.fit_transform(y)
fe=df[["district"]]
array=le.fit_transform(fe)
encode_data=pd.DataFrame(array,columns=fe.columns)

X=pd.concat([x,encode_data],axis=1)
ss=StandardScaler()
X_final=ss.fit_transform(X)

x_train,x_test,y_train,y_test=train_test_split(X_final,y,test_size=0.2,random_state=60)

knc=KNeighborsClassifier(n_neighbors=20)
knc.fit(x_train,y_train)

print("Test Score=> ",knc.score(x_test,y_test))
print("Train Score=> ",knc.score(x_train,y_train))
print("Precision Score=> ",precision_score(y_test,knc.predict(x_test),average="macro"))
print("F1 Score=> ",f1_score(y_test,knc.predict(x_test),average="macro"))
print("Recall Score=> ",recall_score(y_test,knc.predict(x_test),average="macro"))

confusion=confusion_matrix(y_test,knc.predict(x_test))
sns.heatmap(confusion,annot=True)
plt.show()