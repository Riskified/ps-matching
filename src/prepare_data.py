from typing import Any, Dict, List, Union
from pandas import DataFrame, Series, read_csv


class PrepData:
    group_label = Series(dtype=bool)
    group_map = Dict
    target_label = Series(dtype=bool)
    target_map = Dict
    def __init__(self, file_path: str, group: Union[str, List], target: Union[str, List], index_col: str):
        """
            PScorer Class -- creates balanced dataset using propensity score matching
            Parameters
            ----------
            # data : DataFrame
            #     Data with the group variable and the label_col of covariatesto be balanced
            group : str
                The variable that indicates the intervention
            # minority_class_in_group : str
            #     The minority class of the group (the intervention)
        """
        data: DataFrame = read_csv(file_path, index_col=index_col)
        print(f'loaded data with {len(data)} observations')
        self.input: DataFrame = data
        self.create_labels(group, target)
        self.convert_to_categorical()

    @staticmethod
    def get_label_name(label_name: Union[str, List]) -> Union[str, Any]:
        if isinstance(label_name, list):
            return label_name[0], label_name[1]
        else:
            return label_name, None

    @staticmethod
    def get_minority_class(label_col) -> str:
        return label_col.value_counts().tail(1).index[0]

    @staticmethod
    def transform_to_boolean(label: Series, condition: str = None):
        logical_label: Series = label == condition
        print(logical_label.value_counts(normalize=True))
        print(f'The minority class contains {(logical_label == True).sum()} observations')
        return logical_label

    def convert_to_categorical(self):
        cat_cols = self.input.select_dtypes(include='object').columns
        self.input[cat_cols] = self.input[cat_cols].astype('category')

    def create_labels(self, group: Union[str, List[str]], target: Union[str, List[str]]):
        for label in ["group", "target"]:
            name, minority_class = self.get_label_name(locals().get(label))
            condition = minority_class if minority_class is not None else PrepData.get_minority_class(self.input[name])
            label_data: Series = self.transform_to_boolean(self.input[name], condition)
            setattr(self, f"{label}_label", label_data)
            self.input.drop([name], axis=1, inplace=True)
