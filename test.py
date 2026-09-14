from avalanche.training.determinism.rng_manager import RNGManager
from avalanche.training.plugins.checkpoint import CheckpointPlugin, \
    FileSystemCheckpointStorage

from benchmark.plugins import EWCPlugin, LwFPlugin, ERDPlugin, AGEMPlugin
from avalanche.training.plugins import (
    ReplayPlugin,
    SynapticIntelligencePlugin,
    TrainGeneratorAfterExpPlugin,
    BiCPlugin,
)

import logging

from data.SDSDataset import SDSDataset
from avalanche.evaluation.metrics.detection import DetectionMetrics
from avalanche.training.plugins import LRSchedulerPlugin, EvaluationPlugin
import argparse
import torch
from benchmark.detection_examples_utils import split_detection_benchmark

import json
from json import JSONEncoder
from torch import Tensor
from typing import Any
from avalanche.benchmarks.utils.collate_functions import detection_collate_fn
from torch.utils.data import DataLoader
from examples.tvdetection.engine import evaluate
from examples.tvdetection.coco_eval import CocoEvaluator

logging.basicConfig(level=logging.NOTSET)

def load_dataset(ROOT, train=False, transforms=None):
    dataset = SDSDataset(ROOT, train=train, transform=transforms)

    return dataset


def main(args):
    RNGManager.set_random_seeds(1234)
    torch.random.manual_seed(1234)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Deploying plugin: {args.plugins}...")
    print(f"Using device: {device}")

    num_classes = 3 + 1  # N classes + background

    train_mb_size = 1

    train_SDS_1 = load_dataset('/{path_to_files}/train/RedEdge/', train=True)
    test_SDS_1 = load_dataset('/{path_to_files}/val/RedEdge/', train=False)

    train_SDS_2 = load_dataset('/{path_to_files}/train/NIR1/', train=True)
    test_SDS_2 = load_dataset('/{path_to_files}/val/NIR1/', train=False)

    train_SDS_3 = load_dataset('/{path_to_files}/train/NIR2/', train=True)
    test_SDS_3 = load_dataset('/{path_to_files}/val/NIR2/', train=False)

    train_SDS_4 = load_dataset('/{path_to_files}/train/TIR/', train=True)
    test_SDS_4 = load_dataset('/{path_to_files}/val/TIR/', train=False)

    benchmark = split_detection_benchmark(
        train_dataset_lst=[train_SDS_1, train_SDS_2, train_SDS_3, train_SDS_4],
        test_dataset_lst=[test_SDS_1, test_SDS_2, test_SDS_3, test_SDS_4],
        n_classes=num_classes - 1,
    )

    checkpoint_plugin = CheckpointPlugin(
        FileSystemCheckpointStorage(
            directory=f'{args.model_dir}/checkpoints/',
        ),
        map_location=device
    )

    load_strategy, _ = checkpoint_plugin.load_checkpoint_if_exists()
    model = load_strategy.model

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=0.005, momentum=0.9, weight_decay=0.0005
    )

    warmup_factor = 1.0 / 1000
    warmup_iters = min(
        1000, len(benchmark.train_stream[0].dataset) // train_mb_size - 1
    )
    lr_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=warmup_factor, total_iters=warmup_iters
    )

    plugins = [
        LRSchedulerPlugin(
            lr_scheduler,
            step_granularity="iteration",
            first_exp_only=True,
            first_epoch_only=True,
        ),
    ]

    if args.plugins == 'ewc':
        plugins.append(EWCPlugin(ewc_lambda=0.4, mode="separate"))
    elif args.plugins == 'si':
        plugins.append(SynapticIntelligencePlugin(si_lambda=0.0001))
    elif args.plugins == 'lwf':
        plugins.append(LwFPlugin(alpha=1, temperature=2))
    elif args.plugins == 'ERD':
        plugins.append(ERDPlugin(alpha=1, temperature=2))
    elif args.plugins == 'replay':
        plugins.append(ReplayPlugin(mem_size=40))
    elif args.plugins == 'agem':
        plugins.append(AGEMPlugin(patterns_per_experience=30, sample_size=8))

    for idx, dataset in enumerate([test_SDS_1, test_SDS_2, test_SDS_3, test_SDS_4]):
        print("Start training on experience ", idx)
        data_loader = DataLoader(
            dataset, batch_size=train_mb_size, shuffle=False, drop_last=False,
            num_workers=0,
            collate_fn=detection_collate_fn
        )

        eval_result, all_predictions = evaluate(
            model, data_loader, device=device)
        if isinstance(eval_result, CocoEvaluator):
            save_eval_output_to_json(all_predictions, '{}/results_{}.json'.format(args.model_dir, idx + 1))

def save_eval_output_to_json(model_output, json_path):
    print('Saving JSON output to', json_path)
    with open(str(json_path), 'w') as f:
        json.dump(model_output, f, cls=TensorEncoder)
    print('Result correctly saved')

class TensorEncoder(JSONEncoder):
    def __init__(self, **kwargs):
        super(TensorEncoder, self).__init__(**kwargs)

    def default(self, o: Any) -> Any:
        if isinstance(o, Tensor):
            o = o.detach().cpu().tolist()

        return o

def make_det_metrics(detection_only=True):
    if detection_only:
        iou_types = ["bbox"]
    else:
        iou_types = ["bbox", "segm"]

    return DetectionMetrics(
        iou_types=iou_types, default_to_coco=True, summarize_to_stdout=True
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plugins",
        type=str,
        default='naive',
        choices=["naive", "ewc", "si", "lwf", "replay", "agem", "ERD"],
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=''
    )
    parser.add_argument(
        "--checkpoint_at",
        type=int,
        default=-1
    )
    parser.add_argument(
        "--detection_only",
        action="store_true",
        help="Set this flag to ignore the segmentation task",
    )
    main(parser.parse_args())