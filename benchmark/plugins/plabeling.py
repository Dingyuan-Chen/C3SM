import os
import json
import torch
from avalanche.training.plugins.strategy_plugin import SupervisedPlugin

from torch.utils.data import Dataset
from avalanche.benchmarks.utils import AvalancheDataset, make_classification_dataset
from avalanche.benchmarks.utils.collate_functions import detection_collate_fn

from torchvision.ops import nms
is_nms = False

class PseudoLabeledDataset(Dataset):
    def __init__(self, data_list):
        self.data = data_list
        self.targets = [0 for _ in range(len(self.data))]

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

class SSI_PseudoLabelingPlugin(SupervisedPlugin):
    def __init__(self, pseudo_label_json_path, score_threshold=0.5):
        super().__init__()
        self.pseudo_label_json_path = pseudo_label_json_path
        self.score_threshold = score_threshold

    def _load_pseudo_labels(self, task_id):
        file_path = os.path.join(self.pseudo_label_json_path, f"plabels_task{task_id}.json")
        with open(file_path, 'r') as f:
            return json.load(f)

    def before_training_exp(self, strategy, **kwargs):
        task_id = strategy.experience.current_experience
        if task_id > 0:
            pseudo_label_map = self._load_pseudo_labels(task_id)

            new_data = []
            for i in range(len(strategy.adapted_dataset)):
                dd = strategy.adapted_dataset[i]
                image, target = dd[0], dd[1]

                image_id = target["image_id"].item()
                pseudo_anns = pseudo_label_map.get(str(image_id), [])

                resize = [800, 1200]
                org_width = target['org_w'].item()
                org_height = target['org_h'].item()

                pseudo_boxes, pseudo_labels, pseudo_ssi = [], [], []
                if is_nms:
                    pseudo_scores = []

                for ann in pseudo_anns:
                    if ann["score"] >= self.score_threshold:
                        box = ann["bbox"]
                        box[0] = box[0] / (org_width / resize[1])
                        box[1] = box[1] / (org_height / resize[0])
                        box[2] = box[2] / (org_width / resize[1])
                        box[3] = box[3] / (org_height / resize[0])
                        ann["bbox"] = box

                        pseudo_boxes.append(ann["bbox"])
                        pseudo_labels.append(ann["label"])
                        pseudo_ssi.append(ann["SSI"])

                        if is_nms:
                            pseudo_scores.append(ann["score"])

                if pseudo_boxes:
                    pseudo_boxes = torch.tensor(pseudo_boxes, dtype=torch.float32)
                    pseudo_labels = torch.tensor(pseudo_labels, dtype=torch.int64)
                    pseudo_ssi = torch.tensor(pseudo_ssi, dtype=torch.int64)

                    if not is_nms:
                        num_gt = len(target["boxes"])
                        target["boxes"] = torch.cat([target["boxes"], pseudo_boxes], dim=0)
                        target["labels"] = torch.cat([target["labels"], pseudo_labels], dim=0)

                        # SSI
                        target["is_supervised"] = torch.cat([torch.ones(num_gt, dtype=torch.bool), torch.zeros(len(pseudo_boxes), dtype=torch.bool)], dim=0)
                        target["is_pseudo_high"] = torch.cat([torch.ones(num_gt, dtype=torch.bool), 1 - pseudo_ssi], dim=0)
                        target["is_pseudo_low"] = torch.cat([torch.zeros(num_gt, dtype=torch.bool), pseudo_ssi], dim=0)

                    else:
                        pseudo_scores = torch.tensor(pseudo_scores, dtype=torch.float32)
                        gt_boxes = target["boxes"]
                        gt_labels = target["labels"]
                        gt_scores = torch.ones(len(gt_boxes), dtype=torch.float32)

                        all_boxes = torch.cat([gt_boxes, pseudo_boxes], dim=0)
                        all_labels = torch.cat([gt_labels, pseudo_labels], dim=0)
                        all_scores = torch.cat([gt_scores, pseudo_scores], dim=0)

                        all_supervised = torch.cat([torch.ones(len(target["boxes"]), dtype=torch.bool), torch.zeros(len(pseudo_boxes), dtype=torch.bool)], dim=0)
                        all_pseudo_high = torch.cat([torch.ones(len(target["boxes"]), dtype=torch.bool), 1 - pseudo_ssi], dim=0)
                        all_pseudo_low = torch.cat([torch.zeros(len(target["boxes"]), dtype=torch.bool), pseudo_ssi], dim=0)

                        keep_all = []
                        for cls in all_labels.unique():
                            inds = (all_labels == cls).nonzero(as_tuple=True)[0]
                            boxes_cls = all_boxes[inds]
                            scores_cls = all_scores[inds]
                            keep = nms(boxes_cls, scores_cls, iou_threshold=0.7)
                            keep_all.append(inds[keep])

                        if keep_all:
                            keep_all = torch.cat(keep_all)
                            target["boxes"] = all_boxes[keep_all]
                            target["labels"] = all_labels[keep_all]
                            target["score"] = all_scores[keep_all]

                            # SSI
                            target["is_supervised"] = all_supervised[keep_all]
                            target["is_pseudo_high"] = all_pseudo_high[keep_all]
                            target["is_pseudo_low"] = all_pseudo_low[keep_all]


                new_data.append((image, target))

            train_dataset = make_classification_dataset(
                PseudoLabeledDataset(new_data),
                transform_groups=None,
                initial_transform_group="train",
                collate_fn=detection_collate_fn
            )

            strategy.train_dataset = AvalancheDataset(train_dataset)
            strategy.adapted_dataset = strategy.train_dataset
            strategy.experience.dataset = strategy.adapted_dataset
            strategy.make_train_dataloader()