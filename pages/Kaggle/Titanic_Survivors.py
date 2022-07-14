import numpy as np
import pandas as pd
import os
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

class Titanic:

        #load the data
    def load_data(self):
        global data_csv,test_csv
        ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', '..'))
        excel_path = os.path.join(ROOT_DIR, 'data', "Titanic_train.csv")
        excel_path1 = os.path.join(ROOT_DIR, 'data', "Titanic_test.csv")
        data_csv = pd.read_csv(excel_path)
        test_csv = pd.read_csv(excel_path1)
        print(data_csv.head())
        print(data_csv.shape)
        return data_csv

    def data_processing(self):
        global data_csv,test_csv,X,Y
        # Finding the missing value in the dataset
        print(data_csv.isnull().sum())
        print(test_csv.isnull().sum())

        # Drop the missing value from dataset (Cabin)
        data_csv = data_csv.drop("Cabin", axis=1)
        test_csv = test_csv.drop("Cabin", axis=1)

        # Replace missing values in Age with mean
        data_csv["Age"].fillna(data_csv["Age"].mean(), inplace=True)
        test_csv["Age"].fillna(test_csv["Age"].mean(), inplace=True)

        # Replace Embarked values with mode
        print(data_csv["Embarked"].mode())
        data_csv["Embarked"].fillna(data_csv["Embarked"].mode()[0], inplace=True)
        print(test_csv["Embarked"].mode())
        test_csv["Embarked"].fillna(test_csv["Embarked"].mode()[0], inplace=True)

        # Replacing missing Fare value with mean for test data
        test_csv["Fare"].fillna(test_csv["Fare"].mean(), inplace=True)

        # checking for missing values again after replacing
        print(data_csv.isnull().sum())
        print(test_csv.isnull().sum())

        # count the surviours
        print(data_csv["Survived"].value_counts())

        # Encode the label for Sex and Embark
        from sklearn.preprocessing import LabelEncoder

        label = LabelEncoder()
        data_csv['Sex'] = label.fit_transform(data_csv['Sex'])  # 1 male, 0 female
        test_csv['Sex'] = label.fit_transform(test_csv['Sex'])
        print(data_csv['Sex'])
        data_csv['Embarked'] = label.fit_transform(data_csv['Embarked'])  # c-1, Q-2, S-3
        test_csv['Embarked'] = label.fit_transform(test_csv['Embarked'])
        print(data_csv['Embarked'])

        print(data_csv.dtypes)

        # drop unused columns
        data_csv = data_csv.drop(["Name", "Ticket"], axis=1)
        test_csv = test_csv.drop(["Name", "Ticket"], axis=1)

        # seperate Features and Target
        X = data_csv.drop("Survived", axis=1)
        Y = data_csv['Survived']
        return X


    def model_accuracy(self):
        # Seperate Train and test set 80% training and 20% testing
        from sklearn.model_selection import train_test_split
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

        # Scale the data
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler().fit(x_train)
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)


        # Creating function with many machine models

        # Use Logistic Regression
        log = LogisticRegression()
        log.fit(x_train, y_train)

        # Use DecisionTreeClassifier
        tree = DecisionTreeClassifier(criterion='log_loss', random_state=100, splitter='random')
        tree.fit(x_train, y_train)

        # Use RandomForestClassifier
        forest = RandomForestClassifier(n_estimators=10, criterion='gini')
        forest.fit(x_train, y_train)

        # Use Gaussian NB
        gauss = GaussianNB()
        gauss.fit(x_train, y_train)

        # print model score and accuracy score
        from sklearn import metrics

        accuracy = metrics.accuracy_score(y_test, log.predict(x_test))
        print("Logistic Regression accuracy score: ", accuracy)
        accuracy = metrics.accuracy_score(y_test, tree.predict(x_test))
        print("Decision tree accuracy score: ", accuracy)
        accuracy = metrics.accuracy_score(y_test, forest.predict(x_test))
        print("Random Forest accuracy score: ", accuracy)
        accuracy = metrics.accuracy_score(y_test, gauss.predict(x_test))
        print("Gaussian NB accuracy score: ", accuracy)


        # Kfolds accuracy score (considering Decision, Random and Gaussian)
        from sklearn.model_selection import cross_val_score

        score_DR = cross_val_score(DecisionTreeClassifier(), X, Y, cv=10)
        print("\nDecision Tree Average score: ", np.average(score_DR))

        score_DR = cross_val_score(RandomForestClassifier(), X, Y, cv=10)
        print("Random Forest Average score: ", np.average(score_DR))

        score_DR = cross_val_score(GaussianNB(), X, Y, cv=10)
        print("Gaussian NB Average score: ", np.average(score_DR))

        # Selecting Random Forest model and tuning it
        score1 = cross_val_score(RandomForestClassifier(n_estimators=10), X, Y, cv=10)
        score2 = cross_val_score(RandomForestClassifier(n_estimators=20), X, Y, cv=10)
        score3 = cross_val_score(RandomForestClassifier(n_estimators=30), X, Y, cv=10)
        score4 = cross_val_score(RandomForestClassifier(n_estimators=40), X, Y, cv=10)
        print("\nRandom Forest Average score with 10 Estimators: ", np.average(score1))
        print("Random Forest Average score with 20 Estimators: ", np.average(score2))
        print("Random Forest Average score with 30 Estimators: ", np.average(score3))
        print("Random Forest Average score with 40 Estimators: ", np.average(score4))
        return score1,score2,score3,score4

    def realtime_data(self):
        # Selecting correct parameters and fit the model
        prod_model = RandomForestClassifier(n_estimators=30)
        prod_model.fit(X, Y)

        # Predicting the survivors
        y_predict = prod_model.predict(test_csv)
        print(y_predict)

        # output df
        output_df = pd.DataFrame({'PassengerId': test_csv['PassengerId'], 'Survived': y_predict})
        print(output_df.head())
        return output_df

test = Titanic()
test.load_data()
test.data_processing()
test.model_accuracy()
test.realtime_data()


# export
#output_df.to_csv('Survival_Prediction.csv', index=False)

