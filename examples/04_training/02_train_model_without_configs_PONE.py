"""Example of training Model."""

import os
import time
from typing import Any, Dict, List, Optional

from pytorch_lightning.callbacks import EarlyStopping
import torch
from torch.optim.adam import Adam

from graphnet.constants import GRAPHNET_ROOT_DIR, EXAMPLE_OUTPUT_DIR
from graphnet.data.constants import FEATURES, TRUTH
from graphnet.models import StandardModel
from graphnet.models.detector.pone import POne
from graphnet.models.gnn import DynEdge, DynEdgeJINST, ConvNet
from graphnet.models.graph_builders import KNNGraphBuilder
from graphnet.models.task.classification import BinaryClassificationTask
from graphnet.training.callbacks import ProgressBar, PiecewiseLinearLR
from graphnet.training.labels import IsPositive
from graphnet.training.loss_functions import LogCoshLoss, BinaryCrossEntropyLoss
from graphnet.training.utils import make_train_validation_dataloader
from graphnet.utilities.argparse import ArgumentParser
from graphnet.utilities.logging import get_logger

logger = get_logger()

# Constants
features = FEATURES.PONE
truth = TRUTH.PONE


def main(
        path: str,
        pulsemap: str,
        target: str,
        truth_table: str,
        gpus: Optional[List[int]],
        max_epochs: int,
        early_stopping_patience: int,
        batch_size: int,
        num_workers: int,
) -> None:
    """Run example."""
    logger.info(f"features: {features}")
    logger.info(f"truth: {truth}")

    # Configuration
    config: Dict[str, Any] = {
        "path": path,
        "pulsemap": pulsemap,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "target": target,
        "early_stopping_patience": early_stopping_patience,
        "fit": {
            "gpus": gpus,
            "max_epochs": max_epochs,
        },
    }

    t = time.localtime()
    current_time = time.strftime("%Y%m%d_%H%M%S", t)

    archive = os.path.join(
        GRAPHNET_ROOT_DIR,
        "../data/combined_10_20_redistributed/{}_train_model_without_configs_PONE".format(
            current_time
        )
    )
    run_name = "dynedge_{}_example_PONE".format(config["target"])

    (
        training_dataloader,
        validation_dataloader,
    ) = make_train_validation_dataloader(
        config["path"],
        None,
        config["pulsemap"],
        features,
        truth,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        truth_table=truth_table,
        labels={
            'is_event': IsPositive(key='energy')
        }
    )

    # Building model
    detector = POne(
        graph_builder=KNNGraphBuilder(nb_nearest_neighbours=8),
    )
    # gnn = DynEdgeJINST(
    #     nb_inputs=detector.nb_outputs,
    #     layer_size_scale=2
    # )
    # gnn = ConvNet(
    #     nb_inputs=detector.nb_outputs,
    #     nb_intermediate=8,
    #     dropout_ratio=0.3,
    #     nb_outputs=1,
    # )
    gnn = DynEdge(
        nb_inputs=detector.nb_outputs,
        dynedge_layer_sizes=[
            (4, 8),
            (16, 8),
            (16, 8),
            (16, 8)
        ],
        post_processing_layer_sizes=[16,8],
        global_pooling_schemes=["min", "max", "mean", "sum"],
    )
    task = BinaryClassificationTask(
        hidden_size=gnn.nb_outputs,
        target_labels='is_noise',
        loss_function=BinaryCrossEntropyLoss(),
    )
    model = StandardModel(
        detector=detector,
        gnn=gnn,
        tasks=[task],
        optimizer_class=Adam,
        optimizer_kwargs={"lr": 1e-03, "eps": 1e-03},
        scheduler_class=PiecewiseLinearLR,
        scheduler_kwargs={
            "milestones": [
                0,
                len(training_dataloader) / 2,
                len(training_dataloader) * config["fit"]["max_epochs"],
            ],
            "factors": [1e-2, 1, 1e-02],
        },
        scheduler_config={
            "interval": "step",
        },
    )

    # Training model
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=config["early_stopping_patience"],
        ),
        ProgressBar(),
    ]

    model.fit(
        training_dataloader,
        validation_dataloader,
        callbacks=callbacks,
        **config["fit"],
    )

    # Get predictions
    prediction_columns = [config["target"] + "_pred"]
    additional_attributes = [config["target"]]

    results = model.predict_as_dataframe(
        validation_dataloader,
        prediction_columns=prediction_columns,
        additional_attributes=additional_attributes + ["event_no"],
    )

    # Save predictions and model to file
    db_name = path.split("/")[-1].split(".")[0]
    path = os.path.join(archive, db_name, run_name)
    logger.info(f"Writing results to {path}")
    os.makedirs(path, exist_ok=True)

    results.to_csv(f"{path}/results.csv")
    model.save_state_dict(f"{path}/state_dict.pth")
    model.save(f"{path}/model.pth")


if __name__ == "__main__":
    # Parse command-line arguments
    parser = ArgumentParser(
        description="""
Train GNN model without the use of config files.
"""
    )

    parser.add_argument(
        "--path",
        help="Path to dataset file (default: %(default)s)",
        default=f"{GRAPHNET_ROOT_DIR}/../data/combined_10_20_redistributed/graphnet.db",
    )

    parser.add_argument(
        "--pulsemap",
        help="Name of pulsemap to use (default: %(default)s)",
        default="detector_response",
    )

    parser.add_argument(
        "--target",
        help=(
            "Name of feature to use as regression target (default: "
            "%(default)s)"
        ),
        default="is_noise",
    )

    parser.add_argument(
        "--truth-table",
        help="Name of truth table to be used (default: %(default)s)",
        default="mc_truth",
    )

    parser.with_standard_arguments(
        ("gpus", 1),
        ("max-epochs", 16),
        "early-stopping-patience",
        ("batch-size", 8),
        "num-workers",
    )

    args = parser.parse_args()

    main(
        args.path,
        args.pulsemap,
        args.target,
        args.truth_table,
        args.gpus,
        args.max_epochs,
        args.early_stopping_patience,
        args.batch_size,
        args.num_workers,
    )
