from functools import partial
import numpy as np
import torch
from tqdm import tqdm

from neuralpredictors.measures import PoissonLoss
from neuralpredictors.training import early_stopping, MultipleObjectiveTracker
from .utility.nn_helper import set_random_seed

from .utility import measures
from .utility.measures import get_correlations, get_poisson_loss


def train_model(
    model,
    dataloader,
    seed=None,
    loss=None,
    avg_loss=False,
    scale_loss=True,
    stop_function="get_correlations",
    loss_accum_batch_n=None,
    device="cuda",
    verbose=True,
    interval=1,
    patience=5,
    epoch=0,
    lr_init=0.005,
    max_iter=200,
    maximize=True,
    tolerance=1e-6,
    restore_best=True,
    lr_decay_steps=3,
    lr_decay_factor=0.3,
    min_lr=0.0001,
    cb=None,
    track_training=True,
    return_test_score=False,
    **kwargs
):
    """
    Args:
        model: model to be trained
        dataloaders: dataloaders containing the data to train the model with
        seed: random seed
        loss: Criterion module. If None (default), use PoissonLoss
        avg_loss: whether to average (or sum) the loss over a batch. Only used for PoissonLoss instantiation when `loss` is None
        scale_loss: whether to scale the loss according to the size of the dataset
        loss_function: loss function to use
        stop_function: the function (metric) that is used to determine the end of the training in early stopping
        loss_accum_batch_n: number of batches to accumulate the loss over
        device: device to run the training on
        verbose: whether to print out a message for each optimizer step
        interval: interval at which objective is evaluated to consider early stopping
        patience: number of times the objective is allowed to not become better before the iterator terminates
        epoch: starting epoch
        lr_init: initial learning rate
        max_iter: maximum number of training iterations
        maximize: whether to maximize or minimize the objective function
        tolerance: tolerance for early stopping
        restore_best: whether to restore the model to the best state after early stopping
        lr_decay_steps: how many times to decay the learning rate after no improvement
        lr_decay_factor: factor to decay the learning rate with
        min_lr: minimum learning rate
        cb: whether to execute callback function
        track_training: whether to track and print out the training progress
        **kwargs:
    Returns:
    """

    ##### Model training ####################################################################################################
    # check if CUDA is available
    if "cuda" in device and not torch.cuda.is_available():
        print("CUDA is not available! Switching back to CPU")
        device = "cpu"

    model.to(device)
    if seed is not None:
        set_random_seed(seed)
    model.train()

    # instantiate the loss
    if loss is None:
        loss = PoissonLoss(avg=avg_loss)
    criterion = loss
    # set criterion to the target
    criterion.to(device)

    def full_objective(model, dataloader, images, responses, *args):

        loss_scale = (
            np.sqrt(len(dataloader.dataset) / images.shape[0]) if scale_loss else 1.0
        )

        total_loss = (
            loss_scale * criterion(model(images.to(device)), responses.to(device))
            + model.regularizer()
        )

        return total_loss

    stop_closure = partial(
        get_correlations,
        dataloader=dataloader["validation"],
        device=device,
        per_neuron=False,
    )

    n_iterations = len(dataloader["train"])

    optimizer = torch.optim.Adam(model.parameters(), lr=lr_init)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max" if maximize else "min",
        factor=lr_decay_factor,
        patience=patience,
        threshold=tolerance,
        min_lr=min_lr,
        verbose=verbose,
        threshold_mode="abs",
    )

    # set the number of iterations over which you would like to accummulate gradients
    optim_step_count = 1 if loss_accum_batch_n is None else loss_accum_batch_n

    if track_training:
        tracker_dict = dict(
            correlation=partial(
                get_correlations,
                model=model,
                dataloader=dataloader["validation"],
                device=device,
                per_neuron=False,
            ),
            poisson_loss=partial(
                get_poisson_loss,
                model,
                dataloader["validation"],
                device=device,
                per_neuron=False,
            ),
        )
        if hasattr(model, "tracked_values"):
            tracker_dict.update(model.tracked_values)
        tracker = MultipleObjectiveTracker(**tracker_dict)
    else:
        tracker = None

    # train over epochs
    for epoch, val_obj in early_stopping(
        model,
        stop_closure,
        interval=interval,
        patience=patience,
        start=epoch,
        max_iter=max_iter,
        maximize=maximize,
        tolerance=tolerance,
        restore_best=restore_best,
        tracker=tracker,
        scheduler=scheduler,
        lr_decay_steps=lr_decay_steps,
    ):

        # print the quantities from tracker
        if verbose and tracker is not None:
            print("=======================================")
            for key in tracker.log.keys():
                print(key, tracker.log[key][-1], flush=True)

        # executes callback function if passed in keyword args
        if cb is not None:
            cb()

        # train over batches
        optimizer.zero_grad()
        for batch_no, data in tqdm(
            enumerate(dataloader["train"]),
            total=n_iterations,
            desc="Epoch {}".format(epoch),
        ):

            loss = full_objective(model, dataloader["train"], *data)
            loss.backward()
            if (batch_no + 1) % optim_step_count == 0:
                optimizer.step()
                optimizer.zero_grad()

    ##### Model evaluation ####################################################################################################
    model.eval()
    tracker.finalize() if track_training else None

    # Compute avg validation and test correlation
    validation_correlation = get_correlations(
        model, dataloader["validation"], device=device, as_dict=False, per_neuron=False
    )
    test_correlation = get_correlations(
        model, dataloader["test"], device=device, as_dict=False, per_neuron=False
    )

    # return the whole tracker output as a dict
    output = {k: v for k, v in tracker.log.items()} if track_training else {}
    output["validation_corr"] = validation_correlation

    score = (
        np.mean(test_correlation)
        if return_test_score
        else np.mean(validation_correlation)
    )
    return score, output, model.state_dict()
