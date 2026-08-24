# Manuscript Notes

These notes preserve rationale for later paper drafting. They are not
experimental settings or results; the scientific protocol is
[`protocol.md`](protocol.md).

## Fixed Batch and Scale-Specific Peak LR

> We fix the global batch size across model scales and tune only the peak
> learning rate independently for each model. At a fixed training-token budget,
> changing batch size changes the number of optimizer updates and would
> therefore confound model scale with optimization trajectory; fixing the batch
> ensures that all models process the same number of tokens with the same
> number of updates. This follows the controlled-scaling philosophy of
> [Pythia](https://proceedings.mlr.press/v202/biderman23a.html) while using a
> smaller batch appropriate for the substantially smaller
> [MiniPile](https://arxiv.org/abs/2304.08442) corpus. We retune learning rate
> because optimal optimization hyperparameters are scale-dependent, with
> empirical scaling studies showing that the preferred learning rate decreases
> as training compute increases
> ([DeepSeek-AI, 2024](https://arxiv.org/abs/2401.02954)).

## LR Schedule

> We use 1% linear warmup followed by cosine decay to 10% of the peak learning
> rate, matching the established Pythia pretraining schedule. The schedule
> shape is fixed across model scales; only the peak learning rate is tuned.
