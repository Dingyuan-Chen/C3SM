from avalanche.benchmarks import StreamUserDef
from avalanche.benchmarks.scenarios.detection_scenario import (
    DetectionCLScenario,
)
from avalanche.benchmarks.utils import (
    make_classification_dataset,
    classification_subset,
)
from avalanche.benchmarks.utils.collate_functions import detection_collate_fn


def split_detection_benchmark(
    train_dataset_lst,
    test_dataset_lst,
    n_classes: int,
):
    # Note: in future versions of Avalanche, the make_classification_dataset
    # function will be replaced with a more specific function for object 
    # detection datasets.
    train_dataset_avl_lst = []
    for train_dataset in train_dataset_lst:
        train_dataset_avl = make_classification_dataset(
            train_dataset,
            transform_groups=None,
            initial_transform_group="train",
            collate_fn=detection_collate_fn
        )
        train_dataset_avl_lst.append(train_dataset_avl)

    test_dataset_avl_lst = []
    for test_dataset in test_dataset_lst:
        test_dataset_avl = make_classification_dataset(
            test_dataset,
            transform_groups=None,
            initial_transform_group="eval",
            collate_fn=detection_collate_fn
        )
        test_dataset_avl_lst.append(test_dataset_avl)


    train_exps_datasets = []
    for train_dataset_avl in train_dataset_avl_lst:
        idx_range = [i for i in range(len(train_dataset_avl))]
        train_exps_datasets.append(
            classification_subset(train_dataset_avl, indices=idx_range)
        )

    test_exps_datasets = []
    for test_dataset_avl in test_dataset_avl_lst:
        idx_range = [i for i in range(len(test_dataset_avl))]
        test_exps_datasets.append(
            classification_subset(test_dataset_avl, indices=idx_range)
        )

    train_def = StreamUserDef(
        exps_data=train_exps_datasets,
        exps_task_labels=[0 for _ in range(len(train_exps_datasets))],
        is_lazy=False,
    )

    test_def = StreamUserDef(
        exps_data=test_exps_datasets,
        exps_task_labels=[0 for _ in range(len(train_exps_datasets))],
        is_lazy=False,
    )

    return DetectionCLScenario(
        n_classes=n_classes,
        stream_definitions={"train": train_def, "test": test_def},
        complete_test_set_only=False,
    )


__all__ = ["split_detection_benchmark"]
