# Copyright 2023 Lawrence Livermore National Security, LLC and other
# LUAR Project Developers.
#
# SPDX-License-Identifier: Apache-2.0

import argparse

from datasets.pan21_dataset import Pan21Dataset
from datasets.pan22_dataset import Pan22Dataset
from datasets.pan23_dataset import Pan23Dataset
from datasets.pan24_dataset import Pan24Dataset


DNAME_TO_CLASS = {
    "pan21": Pan21Dataset,
    "pan22": Pan22Dataset,
    "pan23": Pan23Dataset,
    "pan24": Pan24Dataset,
}


def get_dataset(
    params: argparse.Namespace,
    split: str,
    only_queries=False,
    only_targets=False
):
    """Returns the appropriate dataset for the selected PAN task."""

    assert split in ["train", "validation", "test"]

    if split == "train":
        return get_train_dataset(params)

    return get_val_or_test_dataset(
        params,
        split,
        only_queries,
        only_targets
    )


def get_train_dataset(
    params: argparse.Namespace
):
    """Returns the training dataset."""

    num_sample_per_author = params.num_sample_per_author
    dataset_class = DNAME_TO_CLASS[params.dataset_name]

    train_dataset = dataset_class(
        params,
        "train",
        num_sample_per_author=num_sample_per_author
    )

    return train_dataset


def get_val_or_test_dataset(
    params,
    split,
    only_queries=False,
    only_targets=False
):
    """Returns the validation or test dataset."""

    dataset_class = DNAME_TO_CLASS[params.dataset_name]

    assert (
        (only_queries == False and only_targets == False)
        or (only_queries ^ only_targets)
    ), "specified both only_queries=True and only_targets=True"

    if only_queries:
        queries = dataset_class(
            params,
            split,
            num_sample_per_author=1,
            is_queries=True
        )
        return queries

    if only_targets:
        targets = dataset_class(
            params,
            split,
            num_sample_per_author=1,
            is_queries=False
        )
        return targets

    queries = dataset_class(
        params,
        split,
        num_sample_per_author=1,
        is_queries=True
    )

    targets = dataset_class(
        params,
        split,
        num_sample_per_author=1,
        is_queries=False
    )

    return queries, targets
