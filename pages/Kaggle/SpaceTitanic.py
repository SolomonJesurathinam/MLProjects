import os
import pandas as pd

class Space_Titanic:

    def spaceTitanic(self):
        # read data
        ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', '..'))
        excel_path = os.path.join(ROOT_DIR, 'data', "space_train.csv")
        excel_path1 = os.path.join(ROOT_DIR, 'data', "space_test.csv")
        data_csv = pd.read_csv(excel_path)
        test_data = pd.read_csv(excel_path1)

        # split Name and Cabin
        data_csv["FirstName"] = data_csv["Name"].str.split(" ", expand=True)[0]
        data_csv["Family"] = data_csv["Name"].str.split(" ", expand=True)[1]

        test_data["FirstName"] = test_data["Name"].str.split(" ", expand=True)[0]
        test_data["Family"] = test_data["Name"].str.split(" ", expand=True)[1]

        data_csv["CabinDeck"] = data_csv["Cabin"].str.split("/", expand=True)[0]
        data_csv["CabinNum"] = data_csv["Cabin"].str.split("/", expand=True)[1]
        data_csv["CabinSide"] = data_csv["Cabin"].str.split("/", expand=True)[2]

        test_data["CabinDeck"] = test_data["Cabin"].str.split("/", expand=True)[0]
        test_data["CabinNum"] = test_data["Cabin"].str.split("/", expand=True)[1]
        test_data["CabinSide"] = test_data["Cabin"].str.split("/", expand=True)[2]


        # Check for null values

        # drop Name and Cabin
        data_csv = data_csv.drop(["Name", "Cabin"], axis=1)
        test_data = test_data.drop(["Name", "Cabin"], axis=1)

        data_csv.info()

        # Change data type of Cabin Number
        data_csv["CabinNum"] = pd.to_numeric(data_csv["CabinNum"])
        test_data["CabinNum"] = pd.to_numeric(test_data["CabinNum"])

        # Seperate numeric and categoric data
        numeric_data = [column for column in data_csv.select_dtypes(["int", "float"])]
        categoric_data = [column for column in data_csv.select_dtypes(exclude=["int", "float"])]
        categoric_data1 = [column for column in test_data.select_dtypes(exclude=["int", "float"])]

        # fill na for numeric data
        for i in numeric_data:
            data_csv[i].fillna(data_csv[i].median(), inplace=True)
            test_data[i].fillna(test_data[i].median(), inplace=True)
        # fill na for categoric data
        for i in categoric_data:
            data_csv[i].fillna(data_csv[i].value_counts().index[0], inplace=True)

        for i in categoric_data1:
            test_data[i].fillna(test_data[i].value_counts().index[0], inplace=True)

        # checking correleation

        # Encoding the categoric data
        from sklearn.preprocessing import OrdinalEncoder
        encoder = OrdinalEncoder()
        data_csv[categoric_data] = encoder.fit_transform(data_csv[categoric_data])
        test_data[categoric_data1] = encoder.fit_transform(test_data[categoric_data1])

        # Scaling the data
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        data_csv[numeric_data] = scaler.fit_transform(data_csv[numeric_data])
        test_data[numeric_data] = scaler.fit_transform(test_data[numeric_data])

        # split X and Y
        X = data_csv.drop("Transported", axis=1)
        Y = data_csv["Transported"]

        # split train and test data
        from sklearn.model_selection import train_test_split
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)

        # Check accuracy score from 3 models
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score
        from xgboost import XGBClassifier

        model = XGBClassifier(learning_rate=0.1, max_depth=5, colsample_bytree=0.8, seed=27)
        model.fit(x_train, y_train)
        xgb_accuracy = accuracy_score(y_test, model.predict(x_test))

        model1 = RandomForestClassifier(n_estimators=100)
        model1.fit(x_train, y_train)
        randomForest_accuracy = accuracy_score(y_test, model1.predict(x_test))

        from sklearn.tree import DecisionTreeClassifier
        model3 = DecisionTreeClassifier()
        model3.fit(x_train, y_train)
        decionTree_accuracy = accuracy_score(y_test, model3.predict(x_test))

        accuracyScore = (xgb_accuracy,randomForest_accuracy,decionTree_accuracy)
        # fit whole data for best model
        model_prod = XGBClassifier()
        model_prod.fit(X, Y)
        #y_predict = pd.Series(model_prod.predict(test_data)).map({0: False, 1: True})
        y_predict = model_prod.predict(test_data)
        # output the values to DF
        test_data1 = pd.read_csv(excel_path1)
        output = pd.DataFrame({"PassengerId": test_data1.PassengerId, "Transported": y_predict})

        #return data
        raw_training_data = pd.read_csv(excel_path)
        raw_testing_data = pd.read_csv(excel_path1)
        processed_data = data_csv

        return raw_training_data,raw_testing_data,processed_data,accuracyScore,output

#obj = Space_Titanic()
#obj.spaceTitanic()






