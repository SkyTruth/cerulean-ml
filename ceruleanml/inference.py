import torch


def save_fastai_model_state_dict_and_tracing(learner, dls, savename, experiment_dir):
    sd = learner.model.state_dict()
    torch.save(
        sd, f"{experiment_dir}/state_dict_{savename}"
    )  # saves state_dict for loading with fastai
    x, _ = dls.one_batch()
    learner.model.cuda()
    learner.model.eval()
    torch.jit.save(
        torch.jit.trace(learner.model, x), f"{experiment_dir}/tracing_{savename}"
    )


def load_tracing_model(savepath):
    tracing_model = torch.jit.load(savepath)
    return tracing_model


def test_tracing_model_one_batch(dls, tracing_model):
    x, _ = dls.one_batch()
    out_batch_logits = tracing_model(x)
    return out_batch_logits


def logits_to_classes(out_batch_logits):
    """returns the confidence scores of the max confident classes
    and an array of max confident class ids.
    """
    probs = torch.nn.functional.softmax(out_batch_logits, dim=1)
    conf, classes = torch.max(probs, 1)
    return conf, classes
