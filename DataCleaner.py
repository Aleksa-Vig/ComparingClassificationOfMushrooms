import os.path as path
from sklearn.preprocessing import LabelEncoder
from typing import Tuple

import pandas as pd

BASE_DIRECTORY = path.abspath(path.dirname('Assignment2'))
DATAFOLDER = path.join(BASE_DIRECTORY, 'Data')
DATAFILE = path.join(DATAFOLDER, 'waveform-+noise.csv')


class DataCleaner:

    def getAttributesAndFeatures(self) -> Tuple[pd.DataFrame, pd.DataFrame]:

        print(BASE_DIRECTORY)
        print(DATAFOLDER)

        if path.exists(DATAFILE):
            print(f'File {DATAFILE} found.... cleaning')
            csvDataFrame: pd.DataFrame = pd.DataFrame(pd.read_csv(DATAFILE))

            # append simple column names
            csvDataFrame.columns = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41']

            # csvDataFrameEncoded = self.encodeDataToNumerical(csvDataFrame)

            # convert non-numerical fields into numerical ones
            # csvDataFrameEncoded = MultiColumnLabelEncoder().fit_transform(csvDataFrame)

            labelFrame, featuresFrame = self.cleanData(csvDataFrame)
            return labelFrame, featuresFrame
        else:
            raise FileNotFoundError(f'File {DATAFILE} not found.')

    def cleanData(self, dataframe: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        labelFrame = dataframe.iloc[:, 40:41]  # Select the 41st column for feature
        featuresFrame = dataframe.iloc[:, :40]  # Select the rest of the columns for attributes
        return labelFrame, featuresFrame

    # def encodeDataToNumerical(self, dataframe: pd.DataFrame) -> pd.DataFrame:
    #
    #     labelsToLoop = dataframe.columns
    #     le = LabelEncoder()
    #
    #     for label in labelsToLoop:
    #         encodedColumnContents = le.fit_transform(dataframe[label])
    #         dataframe.drop(label, axis=1, inplace=True)
    #         dataframe[label] = encodedColumnContents
    #
    #     return dataframe

# if __name__ == "__main__":
#     # pd.set_option('display.max_columns', 50)
#     # pd.set_option('display.max_colwidth', 100)
#     #
#     # featureFrame, attributeFrame = DataCleaner().getAttributesAndFeatures()
#     # print("Statistics on the feature")
#     # print(featureFrame.describe(include='all'))
#     # print("Statistics on the attributes")
#     # print(attributeFrame.describe(include='all'))
