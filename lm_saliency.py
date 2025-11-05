"""
This file is adapted from the original implementation at https://github.com/kayoyin/interpret-lm.git.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch

plt.rcParams["figure.figsize"] = [10, 10]


# Adapted from AllenNLP Interpret and Han et al. 2020
def register_embedding_list_hook(model, embeddings_list):
    def forward_hook(module, inputs, output):
        embeddings_list.append(output.squeeze(0).clone().cpu().detach().numpy())

    if "pythia" in model.config.name_or_path:
        embedding_layer = model.gpt_neox.embed_in
    else:
        embedding_layer = model.model.embed_tokens

    handle = embedding_layer.register_forward_hook(forward_hook)
    return handle


def register_embedding_gradient_hooks(model, embeddings_gradients):
    def hook_layers(module, grad_in, grad_out):
        embeddings_gradients.append(grad_out[0].detach().cpu().numpy())

    if "pythia" in model.config.name_or_path:
        embedding_layer = model.gpt_neox.embed_in
    else:
        embedding_layer = model.model.embed_tokens

    hook = embedding_layer.register_backward_hook(hook_layers)
    return hook


def saliency(
    model,
    input_ids,
    input_mask,
    correct=None,
    foil=None,
    all_possible_foil_ids=None,
):
    # Get model gradients and input embeddings
    torch.enable_grad()
    model.eval()

    # handle handling the embeddings list
    embeddings_list = []
    handle = register_embedding_list_hook(model, embeddings_list)

    # hook handling the embeddings gradients
    embeddings_gradients = []
    hook = register_embedding_gradient_hooks(model, embeddings_gradients)

    if correct is None:
        assert correct is not None, "Correct label must be provided."
        correct = input_ids[-1]

    input_ids = input_ids[:-1]
    input_mask = input_mask[:-1]
    input_ids = torch.unsqueeze(
        torch.tensor(input_ids, dtype=torch.long).to(model.device), 0
    )
    input_mask = torch.unsqueeze(
        torch.tensor(input_mask, dtype=torch.long).to(model.device), 0
    )

    model.zero_grad()
    A = model(input_ids, attention_mask=input_mask)

    if foil is not None and correct != foil:
        if all_possible_foil_ids is not None:
            assert len(all_possible_foil_ids) == 9, (
                "all_possible_foil_ids should have 9 foils."
            )
            assert correct not in all_possible_foil_ids, (
                "Foil should not be in all_possible_foil_ids."
            )

            all_possible_foil_ids_logits_list = []
            for possible_foils in all_possible_foil_ids:
                all_possible_foil_ids_logits_list.append(
                    A.logits.squeeze()[-1][possible_foils]
                )
            stacked_foil_logits = torch.stack(all_possible_foil_ids_logits_list)
            mean_foil_logits = torch.mean(stacked_foil_logits, dim=0)
            (A.logits.squeeze()[-1][correct] - mean_foil_logits).backward()
        else:
            # try:
            (A.logits.squeeze()[-1][correct] - A.logits.squeeze()[-1][foil]).backward()
            # except RuntimeError:
            #     from transformers import AutoTokenizer
            #     tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B")
            #     ipdb.set_trace()
    else:
        (A.logits.squeeze()[-1][correct]).backward()
    handle.remove()
    hook.remove()

    return np.array(embeddings_gradients).squeeze(), np.array(embeddings_list).squeeze()


def input_x_gradient(grads, embds, normalize=False):
    input_grad = np.sum(grads * embds, axis=-1).squeeze()

    if normalize:
        norm = np.linalg.norm(input_grad, ord=1)
        input_grad /= norm

    return input_grad


def l1_grad_norm(grads, normalize=False):
    l1_grad = np.linalg.norm(grads, ord=1, axis=-1).squeeze()

    if normalize:
        norm = np.linalg.norm(l1_grad, ord=1)
        l1_grad /= norm
    return l1_grad


def erasure_scores(
    model, input_ids, input_mask, correct=None, foil=None, remove=False, normalize=False
):
    model.eval()
    if correct is None:
        assert correct is not None, "Correct label must be provided."
        correct = input_ids[-1]
    input_ids = input_ids[:-1]
    input_mask = input_mask[:-1]
    input_ids = torch.unsqueeze(
        torch.tensor(input_ids, dtype=torch.long).to(model.device), 0
    )
    input_mask = torch.unsqueeze(
        torch.tensor(input_mask, dtype=torch.long).to(model.device), 0
    )

    A = model(input_ids, attention_mask=input_mask)
    softmax = torch.nn.Softmax(dim=0)
    logits = A.logits[0][-1]
    probs = softmax(logits)
    if foil is not None and correct != foil:
        base_score = (probs[correct] - probs[foil]).detach().cpu().numpy()
    else:
        base_score = (probs[correct]).detach().cpu().numpy()

    scores = np.zeros(len(input_ids[0]))
    for i in range(len(input_ids[0])):
        if remove:
            input_ids_i = torch.cat(
                (input_ids[0][:i], input_ids[0][i + 1 :])
            ).unsqueeze(0)
            input_mask_i = torch.cat(
                (input_mask[0][:i], input_mask[0][i + 1 :])
            ).unsqueeze(0)
        else:
            input_ids_i = torch.clone(input_ids)
            input_mask_i = torch.clone(input_mask)
            input_mask_i[0][i] = 0

        A = model(input_ids_i, attention_mask=input_mask_i)
        logits = A.logits[0][-1]
        probs = softmax(logits)
        if foil is not None and correct != foil:
            erased_score = (probs[correct] - probs[foil]).detach().cpu().numpy()
        else:
            erased_score = (probs[correct]).detach().cpu().numpy()
        scores[i] = (
            base_score - erased_score
        )  # higher score = lower confidence in correct = more influential input
    if normalize:
        norm = np.linalg.norm(scores, ord=1)
        scores /= norm
    return scores


def exemplar_based_erasure(
    model,
    tokenizer,
    input_ids,
    input_mask,
    correct=None,
    foil=None,
    remove=False,
    normalize=False,
):
    tokens = [tokenizer.decode(i) for i in input_ids[: len(input_ids)]]
    newline_indices = [i for i, token in enumerate(tokens) if token == "\n"]
    model.eval()
    if correct is None:
        assert correct is not None, "Correct label must be provided."
        correct = input_ids[-1]
    input_ids = input_ids[:-1]
    input_mask = input_mask[:-1]
    input_ids = torch.unsqueeze(
        torch.tensor(input_ids, dtype=torch.long).to(model.device), 0
    )
    input_mask = torch.unsqueeze(
        torch.tensor(input_mask, dtype=torch.long).to(model.device), 0
    )
    with torch.no_grad():
        A = model(input_ids, attention_mask=input_mask)
        softmax = torch.nn.Softmax(dim=0)
        logits = A.logits[0][-1]
        probs = softmax(logits)
        if foil is not None and correct != foil:
            base_score = (probs[correct] - probs[foil]).detach().cpu().numpy()
        else:
            base_score = (probs[correct]).detach().cpu().numpy()

        scores = np.zeros(len(newline_indices))
        # scores = np.zeros(len(input_ids[0]))
        for i in range(len(newline_indices)):
            if remove:
                raise NotImplementedError("remove is not implemented yet.")
            else:
                input_ids_i = torch.clone(input_ids)
                input_mask_i = torch.clone(input_mask)
                if i == 0:
                    input_mask_i[0][1 : newline_indices[0] + 1] = 0
                else:
                    input_mask_i[0][
                        newline_indices[i - 1] + 1 : newline_indices[i] + 1
                    ] = 0

            A = model(input_ids_i, attention_mask=input_mask_i)
            logits = A.logits[0][-1]
            probs = softmax(logits)
            if foil is not None and correct != foil:
                erased_score = (probs[correct] - probs[foil]).detach().cpu().numpy()
            else:
                erased_score = (probs[correct]).detach().cpu().numpy()
            scores[i] = (
                base_score - erased_score
            )  # higher score = lower confidence in correct = more influential input
        # print(scores)
        if normalize:
            norm = np.linalg.norm(scores, ord=1)
            scores /= norm
    return scores


def main():
    pass


if __name__ == "__main__":
    main()
