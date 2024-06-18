class dataCleaning():

    def __init__(self, data):
        self.__data = data

    # @property
    # def _constructor(self):
    #    return dataCleaning

    # @property
    # def _constructor_sliced(self):
    #   return SubclassedSeries

    def narrowData(self, startTime):
        # also handing the inf and nan value
        print("-" * 50)
        slicing_data_df = self.__data[self.__data.datetime >= startTime]
        print(slicing_data_df.describe())
        return slicing_data_df.fillna(0).replace([[np.inf], -np.inf], 0)

    @classmethod
    def Normalize(cls, narrowedData, rename='value'):
        dataWinsorized = narrowedData.copy()
        pivot_df = pd.pivot_table(narrowedData, values='value', index='datetime', columns='id', fill_value=0)
        # winsorize can only handle by series
        for col in pivot_df.T.columns:
            pivot_df.T[col] = winsorize(pivot_df.T[col], limits=[0.025, 0.025])
        normalized_pivot_df = ((pivot_df.T - pivot_df.T.mean()) / pivot_df.T.std()).T
        normalized_pivot_df.reset_index(inplace=True)

        return pd.melt(normalized_pivot_df, id_vars=['datetime'], var_name='id', value_name='value').rename(
            columns={"value": rename})
