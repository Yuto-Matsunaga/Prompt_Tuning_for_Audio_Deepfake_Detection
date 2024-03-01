from typing import List, Optional

import torch
from torch import Tensor, nn

EPSILON = 1e-12


class CostWeightGenerator():
    """
    # Instructions
    - TL;DR: beta = 1 - 1e-2, 1 - 1e-3, 1 - 1e-4, and 1 - 1e-5 are
      recommended for UCF101-50-H-W and UCF101-150-H-W, while
      beta = 1 - 1e-2 and 1 - 1e-3 are recommended for HMDB51-79-H-W.
    - More generally, beta = 1 - 1e-2, 1 - 1e-3, 1 - 1e-4, and 1 - 1e-5 are
      recommended for classwise sample sizes < 10k,
      while beta = 1 - 1e-6 and beta = 1e-7 are also recommended for
      classwise sample sizes > 100k.
    - Larger beta leads to more discriminative weights.
    - beta = 1 - 1e-1, 1 - 1e-2, 1 - 1e-3, ..., up to 1 - 1e-7 are allowed;
      beta = 1 - 1e-8 = 0.99999999 is rounded to 1 in float32 calculations.
    - When beta = 1 - 1e-1, the weights are 0.15, 0.1, 0.1, and 0.1
      for classwise sample size 10, 100, 10k, and 100k, respectively;
      thus beta = 1 - 1e-1 is almost meaningless.
    - beta = 1 - 1e-5, 1 - 1e-6, and 1 - 1e-7 give almost the same weights
      for classwise sample sizes less than 10k.

    # Remark
    A difference from the original paper is
    return_w /= torch.sum(return_w) + EPSILON.
    This ensures that return_w is normalized in the batch axis,
    which is not guaranteed in the origianl paper. This normalization could
    make it easy to balance the learning rate and beta.
    """

    def __init__(self, classwise_sample_sizes: List, beta: float, device: str):
        """
        # Args
        - classwise_sample_sizes: A list of integers.
            The length is equal to the number of classes.
        - beta: A float larger than 0. Larger beta leads to more discriminative weights.
            If beta = 1, weights are simply the inverse class frequencies (1 / N_k,
            where N_k is the sample size of class k).
            If beta = -1, weights = [1,1,...,1] (len=num classes).
            This is useful for non-cost-sensitive learning.
        - labels: Optional. If not None, weights will be re-normalized
            in a batchwise manner. The output shape will be (batch,)
        """
        assert beta < 1. - 1e-8 or beta == 1, \
            "beta should be less than 0.9999999 or 1;" + \
            "otherwise we face the loss of trailing digits."

        # Initialize
        self.num_classes = len(classwise_sample_sizes)
        self.beta: Tensor = torch.tensor(
            beta,
            dtype=torch.get_default_dtype(),
            device=device,
            requires_grad=False)
        self.classwise_sample_sizes: Tensor = torch.tensor(
            classwise_sample_sizes,
            dtype=torch.get_default_dtype(),
            device=device,
            requires_grad=False)

        # Calc cost weights
        if beta == -1:
            weights = torch.tensor(
                [1] * self.num_classes,
                dtype=torch.get_default_dtype(),
                device=device,
                requires_grad=False)  # shape = (num_classes, )

        elif beta == 1:
            weights = 1. / (
                self.classwise_sample_sizes +
                EPSILON)  # shape = (num_classes,)

        else:
            weights = (1. - self.beta) / \
                (1. - self.beta ** self.classwise_sample_sizes +
                 EPSILON)  # (num_classes,)

        # Weight normalizatoin (optional) (Not present in [Cui+])
        weights /= weights.sum() + EPSILON
        self.weights = weights  # (num_classes,)

    def get_weights(self) -> Tensor:
        return self.weights  # (num_classes,)

    def __call__(self, labels: Optional[Tensor] = None) -> Tensor:
        """
        # Remark
        A difference from the original paper is
        return_w /= torch.sum(return_w) + EPSILON.
        This ensures that return_w is normalized in the batch axis,
        which is not guaranteed in the origianl paper. This normalization could
        make it easy to balance the learning rate and beta.

        # Args
        - labels: Tensor with shape (batch,) (non-one-hot).

        # Returns
        - return_w: Tensor with shape (batch,). If labels is None,
            self.weights (shape=(num_classes,)) is returned.
        """
        if labels is not None:
            assert labels.dtype == torch.int64
            return_w = torch.gather(
                self.weights, dim=0, index=labels)  # (batch,)
        else:
            return_w = self.weights

        # Batch-wise normalization (optional) (Not present in [Cui+])
        return_w = return_w / (return_w.sum() + EPSILON)

        return return_w


class XentropyLoss(nn.Module):
    """
    RNN-incompatible.

    # Remark
    Cost-sensitive loss weighting is supported,
    following the class-balanced loss:
    [Cui, Yin, et al.
    "Class-balanced loss based on effective number of samples."
    Proceedings of the IEEE/CVF Conference
    on Computer Vision and Pattern Recognition. 2019.]
    (https://arxiv.org/abs/1901.05555).
    """

    def __init__(
            self,
            num_classes: int,
            classwise_sample_sizes: Optional[List] = None,
            beta: float = -1,
            device: str = torch.cuda.current_device(),
            label_smoothing: float = 0.):
        """
        # Args
        - num_classes:
        - classwise_sample_sizes:
        - beta:
        - label_smoothing: A float in [0., 1.].
        """
        assert 0. <= label_smoothing and label_smoothing <= 1.
        super().__init__()

        self.num_classes = num_classes
        self.classwise_sample_sizes = classwise_sample_sizes
        self.beta = beta
        self.device = device
        self.label_smoothing = label_smoothing
        if classwise_sample_sizes is not None:
            self.cwg: Optional[CostWeightGenerator] = CostWeightGenerator(
                classwise_sample_sizes, beta, device)
            self.weights: Optional[Tensor] = self.cwg.get_weights()
        else:
            self.cwg = None
            self.weights = None

        self.loss_fn = nn.BCEWithLogitsLoss(
            pos_weight =self.weights[0]/self.weights[1],
            reduction='mean',
            #
            )

    def forward(self, logits: Tensor, labels: Tensor) -> Tensor:
        """
        Softmax applied in nn.CrossEntropyLoss.

        # Args
        - labels: A Tensor with shape (batch,). Non-one-hot labels.
        - logits: A Tensor with shape (batch, num classes). Logits.

        # Returns
        loss: A float scalar Tensor.
        """
        loss = self.loss_fn(
            input=logits, target=labels)  # scalar

        return loss
