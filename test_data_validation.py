import pandas as pd
def test_data_schema():
       df = pd.read_csv("data.csv")
       expected_columns = {"feature1", "feature2", "label"}
       assert expected_columns.issubset(df.columns)