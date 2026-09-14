import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.rpn import RegionProposalNetwork

is_spec = True

class RPNWithFilteredTargets(RegionProposalNetwork):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._stored_kwargs = kwargs

    def filter_targets(self, targets):
        filtered = []
        for t in targets:
            if "is_supervised" not in t:
                num_gt = len(t["labels"])
                t["is_supervised"] = torch.ones(num_gt, dtype=torch.bool, device=t["labels"].device)

            is_sup = t["is_supervised"]
            if isinstance(is_sup, list):
                is_sup = torch.tensor(is_sup, dtype=torch.bool, device=t["boxes"].device)
            filtered.append({
                "boxes": t["boxes"][is_sup],
                "labels": t["labels"][is_sup]
            })
        return filtered

    def forward(self, images, features, targets=None):
        if self.training and targets is not None:
            targets = self.filter_targets(targets)
        return super().forward(images, features, targets)

class RoIHeadsWithCE(RoIHeads):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if is_spec:
            nchannel = 512
            self.projection_head = nn.Sequential(
                nn.Linear(1024, nchannel),
                nn.BatchNorm1d(nchannel),
                nn.ReLU(),
                nn.Linear(nchannel, nchannel // 2)
            )

    def forward(self, features, proposals, image_shapes, targets=None):
        is_training = self.training

        if is_training and targets is not None:
            for t in targets:
                assert t["boxes"].dtype.is_floating_point
                assert t["labels"].dtype == torch.int64

                num_gt = len(t["labels"])
                if "is_supervised" not in t:
                    t["is_supervised"] = torch.ones(num_gt, dtype=torch.bool, device=t["labels"].device)
                if "is_pseudo_high" not in t:
                    t["is_pseudo_high"] = torch.zeros(num_gt, dtype=torch.bool, device=t["labels"].device)
                if "is_pseudo_low" not in t:
                    t["is_pseudo_low"] = torch.zeros(num_gt, dtype=torch.bool, device=t["labels"].device)

        if is_training:
            proposals, matched_idxs, labels, regression_targets = self.select_training_samples(proposals, targets)

            is_supervised_list = [t["is_supervised"] for t in targets]

            proposal_supervised_masks = []
            for i in range(len(proposals)):
                matched_idx = matched_idxs[i]
                is_sup = is_supervised_list[i]

                mask = matched_idx >= 0
                mask_sup = torch.zeros_like(mask, dtype=torch.bool)
                mask_sup[mask] = is_sup[matched_idx[mask]]
                proposal_supervised_masks.append(mask_sup)

            supervised_mask = torch.cat(proposal_supervised_masks, dim=0)
        else:
            proposals, matched_idxs, labels, regression_targets = proposals, None, None, None
            supervised_mask = None

        box_features = self.box_roi_pool(features, proposals, image_shapes)
        box_features = self.box_head(box_features)
        class_logits, box_regression = self.box_predictor(box_features)

        if is_training:
            labels_cat = torch.cat(labels, dim=0)
            regression_targets_cat = torch.cat(regression_targets, dim=0)

            loss_classifier = F.cross_entropy(class_logits[supervised_mask], labels_cat[supervised_mask])

            sampled_pos_inds_subset = torch.where(labels_cat > 0)[0]
            pos_mask = sampled_pos_inds_subset[supervised_mask[sampled_pos_inds_subset]]
            labels_pos = labels_cat[pos_mask]

            N = class_logits.shape[0]
            box_regression = box_regression.reshape(N, -1, 4)

            loss_box_reg = F.smooth_l1_loss(
                box_regression[pos_mask, labels_pos],
                regression_targets_cat[pos_mask],
                beta=1.0 / 9,
                reduction="sum"
            ) / supervised_mask.sum().clamp(min=1).float()

            if is_spec:
                proj_features = F.normalize(self.projection_head(box_features), dim=-1)
                all_supervised = all([
                    t["is_supervised"].dtype == torch.bool and t["is_supervised"].all()
                    for t in targets
                ])

                if all_supervised:
                    loss_robust = torch.tensor(0.0, device=class_logits.device)
                else:
                    is_pseudo_high_list = [
                        t.get("is_pseudo_high", torch.zeros_like(t["labels"], dtype=torch.bool)).to(class_logits.device)
                        for t in targets]
                    is_pseudo_low_list = [
                        t.get("is_pseudo_low", torch.zeros_like(t["labels"], dtype=torch.bool)).to(class_logits.device)
                        for t in targets]
                    label_list = [t["labels"] for t in targets]
                    matched_idx_list = matched_idxs

                    feat_per_class = defaultdict(list)
                    negative_per_class = defaultdict(list)

                    start = 0
                    for i in range(len(proposals)):
                        matched_idx = matched_idx_list[i]
                        is_high = is_pseudo_high_list[i]
                        is_low = is_pseudo_low_list[i]
                        labels = label_list[i]
                        this_feats = proj_features[start:start + len(proposals[i])]

                        for j in range(len(proposals[i])):
                            mi = matched_idx[j]
                            if mi >= 0:
                                label = labels[mi].item()
                                if is_high[mi]:
                                    feat_per_class[label].append(this_feats[j])
                                elif is_low[mi]:
                                    negative_per_class[label].append(this_feats[j])
                        start += len(proposals[i])

                    anchors, positives, negatives = [], [], []
                    for cls, feats in feat_per_class.items():
                        if len(feats) >= 2:
                            num_triplets = len(feats)
                            anchor_indices = torch.randperm(len(feats))[:num_triplets]

                            for anchor_idx in anchor_indices:
                                anchor = feats[anchor_idx]
                                other_negatives, other_positives = [], []
                                for other_cls, negs in negative_per_class.items():
                                    if other_cls == cls:
                                        other_positives.extend(negs)
                                    if other_cls != cls:
                                        other_negatives.extend(negs)

                                if not other_negatives:
                                    continue
                                negative_tensor = torch.stack(other_negatives)
                                rand_idx = torch.randint(0, len(negative_tensor), (1,))
                                negative = negative_tensor[rand_idx.item()]

                                if not other_positives:
                                    continue
                                positive_tensor = torch.stack(other_positives)
                                rand_idx_pos = torch.randint(0, len(positive_tensor), (1,))
                                positive = positive_tensor[rand_idx_pos.item()]

                                anchors.append(anchor)
                                positives.append(positive)
                                negatives.append(negative)

                    if anchors and positives and negatives:
                        anchor_feats = torch.stack(anchors)
                        positive_feats = torch.stack(positives)
                        negative_feats = torch.stack(negatives)

                        num_neg = len(negative_feats)
                        num_anchor = len(anchor_feats)

                        if num_neg >= num_anchor:
                            neg_idx = torch.randperm(num_neg)[:num_anchor]
                            negative_feats = negative_feats[neg_idx]
                        else:
                            repeat_times = (num_anchor // num_neg) + 1
                            extended_neg = negative_feats[torch.randint(0, num_neg, (repeat_times * num_neg,))]
                            negative_feats = extended_neg[:num_anchor]

                        min_len = min(anchor_feats.size(0), negative_feats.size(0), positive_feats.size(0))
                        if min_len == 0:
                            loss_robust = torch.tensor(0.0, device=class_logits.device)
                        else:
                            anchor_feats = anchor_feats[:min_len]
                            positive_feats = positive_feats[:min_len]
                            negative_feats = negative_feats[:min_len]

                            sim_pos = F.cosine_similarity(anchor_feats, positive_feats, dim=-1)
                            sim_neg = F.cosine_similarity(anchor_feats, negative_feats, dim=-1)

                            target = torch.ones_like(sim_pos)
                            loss_robust = F.margin_ranking_loss(sim_pos, sim_neg, target, margin=1.0)
                    else:
                        loss_robust = torch.tensor(0.0, device=class_logits.device)

                losses = {
                    "loss_classifier": loss_classifier,
                    "loss_box_reg": loss_box_reg,
                    "loss_robust": loss_robust,
                }
            else:
                losses = {
                    "loss_classifier": loss_classifier,
                    "loss_box_reg": loss_box_reg,
                }

            return {}, losses

        else:
            boxes, scores, labels_out = self.postprocess_detections(
                class_logits, box_regression, proposals, image_shapes
            )

            results = []
            for i in range(len(boxes)):
                results.append({
                    "boxes": boxes[i],
                    "labels": labels_out[i],
                    "scores": scores[i],
                })
            return results, {}