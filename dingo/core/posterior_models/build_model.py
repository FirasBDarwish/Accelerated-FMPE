import os
import torch
import torch.nn as nn

from dingo.core.posterior_models.normalizing_flow import NormalizingFlow
from dingo.core.posterior_models.flow_matching import FlowMatching
from dingo.core.posterior_models.score_matching import ScoreDiffusion



def _resolve_device(dev: str | torch.device | None) -> str:
    """Normalize 'device' and gracefully fall back from cuda to cpu if needed."""
    if dev is None:
        return "cuda" if torch.cuda.is_available() else "cpu"
    if isinstance(dev, torch.device):
        dev = dev.type
    if dev == "cuda":
        try:
            # Will raise if no driver / GPUs not visible
            _ = torch.cuda.device_count()
            if torch.cuda.device_count() == 0:
                return "cpu"
            _ = torch.cuda.current_device()
        except Exception:
            return "cpu"
    return dev


# TODO: where to put this to avoid cyclic imports?
def build_model_from_kwargs(filename=None, settings=None, **kwargs):
    """
    Returns the built model (from settings file or rebuild from file). Extracts the relevant arguments (normalizing flow
    or continuous flow) from setting.
    """
    assert filename is not None or settings is not None

    models_dict = {
        "normalizing_flow": NormalizingFlow,
        "flow_matching": FlowMatching,
        "score_matching": ScoreDiffusion,
    }

    # --- determine model type ---
    if filename is not None:
        d = torch.load(filename, map_location="cpu")  # always safe
        type = d["metadata"]["train_settings"]["model"]["type"]
    else:
        type = settings["train_settings"]["model"]["type"]

    if not type.lower() in models_dict:
        raise ValueError("No valid posterior model specified.")

    model_cls = models_dict[type.lower()]

    # --- fold cf/nf kwargs into posterior_kwargs (unchanged logic) ---
    if settings is not None:
        if type.lower() == "normalizing_flow":
            settings["train_settings"]["model"]["posterior_kwargs"].update(
                settings["train_settings"]["model"].get("nf_kwargs", {})
            )
        if type.lower() in ["flow_matching", "score_matching"]:
            settings["train_settings"]["model"]["posterior_kwargs"].update(
                settings["train_settings"]["model"].get("cf_kwargs", {})
            )
        settings["train_settings"]["model"].pop("nf_kwargs", None)
        settings["train_settings"]["model"].pop("cf_kwargs", None)

    # --- new: robust device handling ---
    # Accept 'device' from kwargs (eval.py passes it). Resolve and enforce.
    requested_device = kwargs.pop("device", None)
    device = _resolve_device(requested_device)
    # If we're going to run on CPU, hide GPUs proactively (prevents surprise CUDA in libs)
    if device == "cpu":
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

    # Build the model (it may wrap DP inside its init)
    model = model_cls(model_filename=filename, metadata=settings, **kwargs)

    # Force the network and internal attribute onto the resolved device
    try:
        model.network_to_device(device)  # DINGO helper if available
    except Exception:
        model.network.to(device)

    # Critical: keep the attribute that downstream code uses in sync
    model.device = torch.device(device)

    # If we are on CPU, unwrap DataParallel to avoid cuda tensors getting created later
    if device == "cpu" and isinstance(getattr(model, "network", None), nn.DataParallel):
        model.network = model.network.module

    return model


def autocomplete_model_kwargs(
    model_kwargs, data_sample=None, input_dim=None, context_dim=None
):
    """
    Autocomplete the model kwargs from train_settings and data_sample from
    the dataloader:
    (*) set input dimension of embedding net to shape of data_sample[1]
    (*) set dimension of nsf parameter space to len(data_sample[0])
    (*) set added_context flag of embedding net if required for gnpe proxies
    (*) set context dim of nsf to output dim of embedding net + gnpe proxy dim

    :param train_settings: dict
        train settings as loaded from .yaml file
    :param data_sample: list
        Sample from dataloader (e.g., wfd[0]) used for autocomplection.
        Should be of format [parameters, GW data, gnpe_proxies], where the
        last element is only there is gnpe proxies are required.
    :return: model_kwargs: dict
        updated, autocompleted model_kwargs
    """

    # If provided, extract settings from the data sample. Otherwise, use provided kwargs. Since input_dim always needs
    # to be provided, we can use this to verify that they are mutually exclusive.
    assert (
        data_sample is not None
        and input_dim is None
        or data_sample is None
        and input_dim is not None
    )

    if data_sample is not None:
        # set input dims from ifo_list and domain information
        model_kwargs["embedding_kwargs"]["input_dims"] = list(data_sample[1].shape)
        # set dimension of parameter space of nsf
        model_kwargs["posterior_kwargs"]["input_dim"] = len(data_sample[0])
        # set added_context flag of embedding net if gnpe proxies are required
        # set context dim of nsf to output dim of embedding net + gnpe proxy dim
        try:
            gnpe_proxy_dim = len(data_sample[2])
            model_kwargs["embedding_kwargs"]["added_context"] = True
            model_kwargs["posterior_kwargs"]["context_dim"] = (
                model_kwargs["embedding_kwargs"]["output_dim"] + gnpe_proxy_dim
            )
        except IndexError:
            model_kwargs["embedding_kwargs"]["added_context"] = False
            model_kwargs["posterior_kwargs"]["context_dim"] = model_kwargs[
                "embedding_kwargs"
            ]["output_dim"]
    else:
        model_kwargs["posterior_kwargs"]["input_dim"] = input_dim
        model_kwargs["posterior_kwargs"]["context_dim"] = context_dim

    # return model_kwargs
