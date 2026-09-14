import os

from avalanche.evaluation.metrics import (
    timing_metrics,
    accuracy_metrics,
    loss_metrics,
)
from avalanche.training.determinism.rng_manager import RNGManager
from avalanche.training.plugins.checkpoint import CheckpointPlugin, \
    FileSystemCheckpointStorage
from avalanche.logging import InteractiveLogger, TensorboardLogger, TextLogger

from benchmark.plugins import EWCPlugin, LwFPlugin, ERDPlugin, AGEMPlugin, BiCPlugin, ReplayPlugin
from avalanche.training.plugins import (
    SynapticIntelligencePlugin,
    TrainGeneratorAfterExpPlugin,
)

import logging

from data.SDSDataset import SDSDataset
from benchmark.naive_object_detection import ObjectDetectionTemplate

from avalanche.evaluation.metrics.detection import DetectionMetrics
from avalanche.training.plugins import LRSchedulerPlugin, EvaluationPlugin
import argparse
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from benchmark.detection_examples_utils import split_detection_benchmark

from torch.utils.data import DataLoader
from avalanche.benchmarks.utils.data_loader import detection_collate_fn
from benchmark.plugins import PseudoLabelingPlugin, KmeansPseudoLabelingPlugin, SSI_PseudoLabelingPlugin

import json
from module.detector import RoIHeadsWithCE, RPNWithFilteredTargets
import math


def load_dataset(ROOT, train=False, transforms=None):
    dataset = SDSDataset(ROOT, train=train, transform=transforms)

    return dataset

def generate_pseudo_labels(model, dataloader, save_path, threshold=0.7, device='cuda'):
    model.eval().to(device)
    pseudo_labels = {}

    if save_path.endswith("1.json"):
        txt_path = "/{path_to_files}/train/NIR1/NIR1_parse.txt"
        json_path = "/{path_to_files}/train/NIR1/truth.json"
    elif save_path.endswith("2.json"):
        txt_path = "/{path_to_files}/train/NIR2/NIR2_parse.txt"
        json_path = "/{path_to_files}/train/NIR2/truth.json"
    else:
        txt_path = "/{path_to_files}/train/TIR/TIR_parse.txt"
        json_path = "/{path_to_files}/train/TIR/truth.json"

    SSI = {}
    with open(txt_path, "r") as f:
        for line in f:
            text, num = line.strip().split()
            SSI[text] = float(num)

    with open(json_path, "r") as f:
        gt = json.load(f)
    id2name = {img["id"]: img["file_name"] for img in gt["images"]}

    for dd in dataloader:
        images, targets = dd[0], dd[1]
        images = [img.to(device) for img in images]
        outputs = model(images)

        resize = [800, 1200]
        org_width = targets[0]['org_w'].item()
        org_height = targets[0]['org_h'].item()

        for output, target in zip(outputs, targets):
            img_id = target["image_id"].item()
            boxes = output["boxes"].detach().cpu()
            # -------- resize ---------
            if len(boxes.size()) > 0:
                box = boxes
                box[:, 0] = box[:, 0] * (org_width / resize[1])
                box[:, 1] = box[:, 1] * (org_height / resize[0])
                box[:, 2] = box[:, 2] * (org_width / resize[1])
                box[:, 3] = box[:, 3] * (org_height / resize[0])
                boxes = box

            labels = output["labels"].detach().cpu()
            scores = output["scores"].detach().cpu()

            pseudo = []
            for box, label, score in zip(boxes, labels, scores):
                if score.item() >= threshold:
                    pseudo.append({
                        "bbox": box.tolist(),
                        "label": label.item(),
                        "score": score.item(),
                        "SSI": SSI[id2name[img_id]]
                    })

            if pseudo:
                pseudo_labels[img_id] = pseudo

    with open(save_path, "w") as f:
        json.dump(pseudo_labels, f)

class TaskSchedulerController:
    def __init__(self):
        self.task_id = 0
        self.T = 500

def build_cosine_lambda_fn(controller, eta_max_base=0.005, eta_min=1e-5, gamma=0.8, warmup_iters=1000):
    def cosine_fn(iteration):
        t = controller.task_id
        T = controller.T
        eta_max_t = eta_max_base

        if t == 0 and iteration < warmup_iters:
            alpha = iteration / warmup_iters
            eta = eta_min + alpha * (eta_max_t - eta_min)
        else:
            cos_inner = math.pi * (iteration % T) / T
            eta = eta_min + 0.5 * (eta_max_t - eta_min) * (1 + math.cos(cos_inner))

        return eta / eta_max_base
    return cosine_fn


def main(args):
    RNGManager.set_random_seeds(1234)
    torch.random.manual_seed(1234)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Deploying plugin: {args.plugins}...")
    print(f"Using device: {device}")

    num_classes = 3 + 1  # N classes + background

    train_mb_size = 4
    val_mb_size = 1
    train_epochs = 25

    is_plabel = True

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

    # load a model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        pretrained=True
    )

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features, num_classes
    )

    model.rpn = RPNWithFilteredTargets(
        anchor_generator=model.rpn.anchor_generator,
        head=model.rpn.head,
        fg_iou_thresh=model.rpn.proposal_matcher.high_threshold,
        bg_iou_thresh=model.rpn.proposal_matcher.low_threshold,
        batch_size_per_image=256,
        positive_fraction=0.5,
        pre_nms_top_n={'training': 2000, 'testing': 1000},
        post_nms_top_n={'training': 2000, 'testing': 1000},
        nms_thresh=model.rpn.nms_thresh
    )

    model.roi_heads = RoIHeadsWithCE(
        box_roi_pool=model.roi_heads.box_roi_pool,
        box_head=model.roi_heads.box_head,
        box_predictor=model.roi_heads.box_predictor,
        fg_iou_thresh=model.roi_heads.proposal_matcher.high_threshold,
        bg_iou_thresh=model.roi_heads.proposal_matcher.low_threshold,
        batch_size_per_image=512,
        positive_fraction=0.25,
        bbox_reg_weights=None,
        score_thresh=model.roi_heads.score_thresh,
        nms_thresh=model.roi_heads.nms_thresh,
        detections_per_img=model.roi_heads.detections_per_img,
    )
    model = model.to(device)

    checkpoint_plugin = CheckpointPlugin(
        FileSystemCheckpointStorage(
            directory=f'{args.model_dir}/checkpoints/',
        ),
        map_location=device
    )
    strategy, initial_exp = checkpoint_plugin.load_checkpoint_if_exists()

    if strategy is None:
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(
            params, lr=0.005, momentum=0.9, weight_decay=0.0005
        )

        controller = TaskSchedulerController()
        warmup_iters = min(1000, len(benchmark.train_stream[0].dataset) // train_mb_size - 1)
        cosine_fn = build_cosine_lambda_fn(controller, warmup_iters=warmup_iters)
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=cosine_fn)

        if is_plabel:
            plugins = [
                checkpoint_plugin,
                LRSchedulerPlugin(
                    lr_scheduler,
                    step_granularity="iteration",
                    first_exp_only=False,
                    first_epoch_only=False
                ),

                SSI_PseudoLabelingPlugin(
                    pseudo_label_json_path=args.model_dir,
                    score_threshold=0.1
                )
            ]
        else:
            plugins = [
                checkpoint_plugin,
                LRSchedulerPlugin(
                    lr_scheduler,
                    step_granularity="iteration",
                    first_exp_only=True,
                    first_epoch_only=True,
                )
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
            plugins.append(AGEMPlugin(patterns_per_experience=10, sample_size=8))

        os.makedirs(f'{args.model_dir}/checkpoints/',
                    exist_ok=True)
        loggers = [
            TextLogger(
                open(f'{args.model_dir}/log.txt', 'w')),
            InteractiveLogger(),
            TensorboardLogger(f'{args.model_dir}/')
        ]

        eval_plugin = EvaluationPlugin(
            timing_metrics(epoch=True),
            loss_metrics(epoch_running=True),
            make_det_metrics(detection_only=True),
            loggers=loggers,
        )

        strategy = ObjectDetectionTemplate(
            model=model,
            optimizer=optimizer,
            train_mb_size=train_mb_size,
            train_epochs=train_epochs,
            eval_mb_size=val_mb_size,
            device=device,
            plugins=plugins,
            evaluator=eval_plugin,
        )

    results = []
    for experience in benchmark.train_stream[initial_exp:]:
        print("Start training on experience ", experience.current_experience)
        strategy.train(experience)

        if is_plabel:
            if experience.current_experience + 1 < len(benchmark.train_stream):
                next_dataset = benchmark.train_stream[experience.current_experience + 1].dataset
                dataloader = DataLoader(
                    next_dataset,
                    num_workers=0,
                    batch_size=1,
                    pin_memory=True,
                    collate_fn=detection_collate_fn,
                )
                generate_pseudo_labels(
                    model=strategy.model,
                    dataloader=dataloader,
                    save_path=os.path.join(args.model_dir, f"plabels_task{experience.current_experience + 1}.json"),
                    threshold=0.3
                )

        results.append(strategy.eval(benchmark.test_stream[-1:]))


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
