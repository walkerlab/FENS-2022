import warnings
import numpy as np
import torch
from neuralpredictors.measures import corr
from neuralpredictors.training import eval_state, device_state
import types
import contextlib


def model_predictions_repeats(
    model, dataloader, device="cpu", broadcast_to_target=False
):
    """
    Computes model predictions for a dataloader that yields batches with identical inputs along the first dimension.
    Unique inputs will be forwarded only once through the model
    Returns:
        target: ground truth, i.e. neuronal firing rates of the neurons as a list: [num_images][num_reaps, num_neurons]
        output: responses as predicted by the network for the unique images. If broadcast_to_target, returns repeated
                outputs of shape [num_images][num_reaps, num_neurons] else (default) returns unique outputs of shape [num_images, num_neurons]
    """

    target = []
    unique_images = torch.empty(0)
    for images, responses in dataloader:
        if len(images.shape) == 5:
            images = images.squeeze(dim=0)
            responses = responses.squeeze(dim=0)

        assert torch.all(
            torch.eq(
                images[
                    -1,
                ],
                images[
                    0,
                ],
            )
        ), "All images in the batch should be equal"
        unique_images = torch.cat(
            (
                unique_images,
                images[
                    0:1,
                ],
            ),
            dim=0,
        )
        target.append(responses.detach().cpu().numpy())

    # Forward unique images once:
    with eval_state(model) if not isinstance(
        model, types.FunctionType
    ) else contextlib.nullcontext():
        with device_state(model, device) if not isinstance(
            model, types.FunctionType
        ) else contextlib.nullcontext():
            output = model(unique_images.to(device)).detach().cpu()

    output = output.numpy()

    if broadcast_to_target:
        output = [np.broadcast_to(x, target[idx].shape) for idx, x in enumerate(output)]

    return target, output


def model_predictions(model, dataloader, device="cpu"):
    """
    computes model predictions for a given dataloader and a model
    Returns:
        target: ground truth, i.e. neuronal firing rates of the neurons
        output: responses as predicted by the network
    """

    target, output = torch.empty(0), torch.empty(0)
    for images, responses in dataloader:
        with torch.no_grad():
            with device_state(model, device) if not isinstance(
                model, types.FunctionType
            ) else contextlib.nullcontext():
                output = torch.cat(
                    (output, (model(images.to(device)).detach().cpu())), dim=0
                )
            target = torch.cat((target, responses.detach().cpu()), dim=0)

    return target.numpy(), output.numpy()


def get_avg_correlations(model, dataloader, device="cpu", per_neuron=True, **kwargs):
    """
    Returns correlation between model outputs and average responses over repeated trials
    """
    if "test" in dataloader:
        dataloader = dataloader["test"]

    # Compute correlation with average targets
    target, output = model_predictions_repeats(
        dataloader=dataloader, model=model, device=device, broadcast_to_target=False
    )
    target_mean = np.array([t.mean(axis=0) for t in target])
    correlation = corr(target_mean, output, axis=0)

    # Check for nans
    if np.any(np.isnan(correlation)):
        warnings.warn(
            "{}% NaNs , NaNs will be set to Zero.".format(
                np.isnan(correlation).mean() * 100
            )
        )
    correlation[np.isnan(correlation)] = 0

    if not per_neuron:
        correlation = np.mean(correlation)
    return correlation


def get_correlations(model, dataloader, device="cpu", per_neuron=True, **kwargs):
    """
    Returns correlation between model outputs and responses
    """
    with eval_state(model) if not isinstance(
        model, types.FunctionType
    ) else contextlib.nullcontext():
        target, output = model_predictions(
            dataloader=dataloader, model=model, device=device
        )
        correlations = corr(target, output, axis=0)

        if np.any(np.isnan(correlations)):
            warnings.warn(
                "{}% NaNs , NaNs will be set to Zero.".format(
                    np.isnan(correlations[k]).mean() * 100
                )
            )
        correlations[np.isnan(correlations)] = 0
    if not per_neuron:
        correlations = np.mean(correlations)
    return correlations


def get_poisson_loss(model, dataloader, device="cpu", per_neuron=True, eps=1e-12):
    with eval_state(model) if not isinstance(
        model, types.FunctionType
    ) else contextlib.nullcontext():
        target, output = model_predictions(
            dataloader=dataloader, model=model, device=device
        )
        loss = output - target * np.log(output + eps)
        poisson_loss = np.sum(loss, axis=0)
    if not per_neuron:
        poisson_loss = np.sum(poisson_loss)
    return poisson_loss


def get_repeats(dataloader, min_repeats=2):
    # save the responses of all neuron to the repeats of an image as an element in a list
    repeated_inputs = []
    repeated_outputs = []
    for inputs, outputs in dataloader:
        if len(inputs.shape) == 5:
            inputs = np.squeeze(inputs.cpu().numpy(), axis=0)
            outputs = np.squeeze(outputs.cpu().numpy(), axis=0)
        else:
            inputs = inputs.cpu().numpy()
            outputs = outputs.cpu().numpy()
        r, n = outputs.shape  # number of frame repeats, number of neurons
        if (
            r < min_repeats
        ):  # minimum number of frame repeats to be considered for oracle, free choice
            continue
        assert np.all(
            np.abs(np.diff(inputs, axis=0)) == 0
        ), "Images of oracle trials do not match"
        repeated_inputs.append(inputs)
        repeated_outputs.append(outputs)
    return np.array(repeated_inputs), np.array(repeated_outputs)


def get_oracles(dataloader, per_neuron=True):
    _, outputs = get_repeats(dataloader)
    oracle = compute_oracle_corr(np.array(outputs))
    if not per_neuron:
        oracle = np.mean(oracle)
    return oracle


def get_oracles_corrected(dataloader, per_neuron=True):
    _, outputs = get_repeats(dataloader)
    oracle = compute_oracle_corr_corrected(np.array(outputs))
    if not per_neuron:
        oracle = np.mean(oracle)
    return oracle


def compute_oracle_corr_corrected(repeated_outputs):
    if len(repeated_outputs.shape) == 3:
        var_noise = repeated_outputs.var(axis=1).mean(0)
        var_mean = repeated_outputs.mean(axis=1).var(0)
    else:
        var_noise, var_mean = [], []
        for repeat in repeated_outputs:
            var_noise.append(repeat.var(axis=0))
            var_mean.append(repeat.mean(axis=0))
        var_noise = np.mean(np.array(var_noise), axis=0)
        var_mean = np.var(np.array(var_mean), axis=0)
    return var_mean / np.sqrt(var_mean * (var_mean + var_noise))


def compute_oracle_corr(repeated_outputs):
    if len(repeated_outputs.shape) == 3:
        _, r, n = repeated_outputs.shape
        oracles = (
            (repeated_outputs.mean(axis=1, keepdims=True) - repeated_outputs / r)
            * r
            / (r - 1)
        )
        if np.any(np.isnan(oracles)):
            warnings.warn(
                "{}% NaNs when calculating the oracle. NaNs will be set to Zero.".format(
                    np.isnan(oracles).mean() * 100
                )
            )
        oracles[np.isnan(oracles)] = 0
        return corr(oracles.reshape(-1, n), repeated_outputs.reshape(-1, n), axis=0)
    else:
        oracles = []
        for outputs in repeated_outputs:
            r, n = outputs.shape
            # compute the mean over repeats, for each neuron
            mu = outputs.mean(axis=0, keepdims=True)
            # compute oracle predictor
            oracle = (mu - outputs / r) * r / (r - 1)

            if np.any(np.isnan(oracle)):
                warnings.warn(
                    "{}% NaNs when calculating the oracle. NaNs will be set to Zero.".format(
                        np.isnan(oracle).mean() * 100
                    )
                )
                oracle[np.isnan(oracle)] = 0

            oracles.append(oracle)
        return corr(np.vstack(repeated_outputs), np.vstack(oracles), axis=0)
