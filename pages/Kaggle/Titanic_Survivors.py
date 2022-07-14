import pandas as pd
import os

class Titanic:

        #load the data
    def load_data(self):
        global data_csv,test_csv,excel_path1
        ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', '..'))
        excel_path = os.path.join(ROOT_DIR, 'data', "Titanic_train.csv")
        excel_path1 = os.path.join(ROOT_DIR, 'data', "Titanic_test.csv")
        data_csv = pd.read_csv(excel_path)
        test_csv = pd.read_csv(excel_path1)
        return data_csv,test_csv

    def data_processing(self):
        global data_csv,test_csv
        # dropping Cabin
        data_csv = data_csv.drop("Cabin", axis=1)
        test_csv = test_csv.drop("Cabin", axis=1)

        # fill nan with mean for Age and mode for Ticket
        data_csv["Age"].fillna(data_csv["Age"].mean(), inplace=True)
        test_csv["Age"].fillna(test_csv["Age"].mean(), inplace=True)
        test_csv["Fare"].fillna(0, inplace=True)

        # fill na for categorical value
        data_csv["Embarked"].fillna(data_csv["Embarked"].mode()[0], inplace=True)
        test_csv["Embarked"].fillna(test_csv["Embarked"].mode()[0], inplace=True)

        # split Family Name and drop Name
        data_csv["FamilyName"] = data_csv["Name"].str.split(",", expand=True)[0]
        data_csv = data_csv.drop("Name", axis=1)

        test_csv["FamilyName"] = test_csv["Name"].str.split(",", expand=True)[0]
        test_csv = test_csv.drop("Name", axis=1)

        # Custom function
        def custom_split(str):
            if str[0].isnumeric():
                return str
            else:
                a = 0
                i = len(str)
                while (i > 1):
                    if str[i - 1] == " ":
                        a = len(str) - i + 1
                        break
                    i = i - 1
                str1 = str[:-a:-1]
                return str1[::-1]

        # Splitting Ticket to get the numeric values
        i = 0
        while (i < len(data_csv["Ticket"])):
            data_csv.loc[i, "Ticket"] = custom_split(data_csv["Ticket"][i])
            i = i + 1
        data_csv["Ticket"] = data_csv["Ticket"].replace("INE", "0")

        # convert to numeric type
        data_csv["Ticket"] = pd.to_numeric(data_csv["Ticket"])

        j = 0
        while (j < len(test_csv["Ticket"])):
            test_csv.loc[j, "Ticket"] = custom_split(test_csv["Ticket"][j])
            j = j + 1

        # convert to numeric type
        data_csv["Ticket"] = pd.to_numeric(data_csv["Ticket"])
        test_csv["Ticket"] = pd.to_numeric(test_csv["Ticket"])

        #data_csv.info()

        # check correleation
        #print(data_csv.corr())

        # split numeric and categoricl data
        numeric_data = [column for column in data_csv.select_dtypes(["int", "float"])]
        categoric_data = [column for column in data_csv.select_dtypes(exclude=["int", "float"])]

        categoric_data1 = [column for column in test_csv.select_dtypes(exclude=["int", "float"])]

        # Encoding categoric data
        from sklearn.preprocessing import OrdinalEncoder
        label = OrdinalEncoder()
        data_csv[categoric_data] = label.fit_transform(data_csv[categoric_data])
        test_csv[categoric_data1] = label.fit_transform(test_csv[categoric_data1])

        # Scaling the data
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        numeric_data.remove("Survived")
        data_csv[numeric_data] = scaler.fit_transform(data_csv[numeric_data])
        test_csv[numeric_data] = scaler.fit_transform(test_csv[numeric_data])

        # check correleation
        #print(data_csv.corr())
        return data_csv

    def model_accuracy(self):
        # split X and Y
        global forest_model
        X = data_csv.drop("Survived", axis=1)
        Y = data_csv["Survived"]

        from sklearn.model_selection import train_test_split
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)

        # Check accuracy of 3 models
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score
        from xgboost import XGBClassifier
        from sklearn.tree import DecisionTreeClassifier

        forest_model = RandomForestClassifier()
        forest_model.fit(x_train, y_train)
        forest_accuracy = accuracy_score(y_test, forest_model.predict(x_test))

        #xgb_model = XGBClassifier(learning_rate=0.1, max_depth=5, colsample_bytree=0.8, seed=27)
        #xgb_model = XGBClassifier()
        #xgb_model.fit(x_train, y_train)
        #xgb_accuracy = accuracy_score(y_test, xgb_model.predict(x_test))
        xgb_accuracy = 100

        decision_model = DecisionTreeClassifier()
        decision_model.fit(x_train, y_train)
        decision_accuracy = accuracy_score(y_test, decision_model.predict(x_test))
        return forest_accuracy, xgb_accuracy, decision_accuracy

    def realtime_data(self):
        y_predict = forest_model.predict(test_csv)
        test_csv1 = pd.read_csv(excel_path1)
        outputDF = pd.DataFrame({'PassengerId': test_csv1['PassengerId'], 'Survived': y_predict})
        return outputDF

test = Titanic()
test.load_data()
test.data_processing()
test.model_accuracy()
test.realtime_data()


# export
#output_df.to_csv('Survival_Prediction.csv', index=False)

