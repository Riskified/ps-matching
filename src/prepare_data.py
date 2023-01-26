from typing import Any, List, Union
from pandas import DataFrame, Series, read_csv


class PrepData:
    group_label = Series(dtype=bool)
    target_label = Series(dtype=bool)

    def __init__(self, file_path: str, group: Union[str, List], target: Union[str, List], index_col: str):
        data: DataFrame = read_csv(file_path, index_col=index_col)
        print(f'loaded data with {len(data)} observations')
        self.input: DataFrame = data
        self.create_labels(group, target)

    @staticmethod
    def get_condition(label_name: Union[str, List]) -> Union[str, Any]:
        if isinstance(label_name, list):
            return label_name[0], label_name[1]
        else:
            return label_name, None

    @staticmethod
    def get_minority_class(label_col) -> str:
        return label_col.value_counts().tail(1).index[0]

    @staticmethod
    def transform_to_boolean(label: Series, minority_class: str = None):
        condition = minority_class if minority_class is not None else PrepData.get_minority_class(label)
        logical_label: Series = label == condition
        print(logical_label.value_counts(normalize=True))
        print(f'The minority class contains {(logical_label == True).sum()} observations')
        return logical_label

    def create_labels(self, group: Union[str, List[str]], target: Union[str, List[str]]):
        for label in ["group", "target"]:
            name, minority_class = self.get_condition(locals().get(label))
            label_data: Series = self.transform_to_boolean(self.input[name], minority_class)
            setattr(self, f"{label}_label", label_data)
            self.input.drop([name], axis=1, inplace=True)
