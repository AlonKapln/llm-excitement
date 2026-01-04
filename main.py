import torch
from tqdm import tqdm

from model_loading import load_gemma_model_and_tokenizer, load_gemma_scope_sae
from load_dataset import load_dataset_with_feedback


def gather_residual_activations(model, target_layer, inputs) -> torch.Tensor:
    target_act = torch.Tensor([])

    def gather_target_act_hook(mod, inputs, outputs):
        nonlocal target_act  # make sure we can modify the target_act from the outer scope
        target_act = outputs[0]
        return outputs

    handle = model.model.layers[target_layer].register_forward_hook(gather_target_act_hook)
    _ = model.forward(**inputs)
    handle.remove()
    return target_act

if __name__ == "__main__":
    torch.set_grad_enabled(False)  # avoid blowing up mem
    torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Example usage of loading model and tokenizer
    tokenizer, model = load_gemma_model_and_tokenizer(model_size="1b", instructions_tuned=True)

    # Example usage of loading SAE
    sae = load_gemma_scope_sae(model_size="1b", instructions_tuned=True).to(model.device)

    # Example usage of loading dataset with feedback
    dataset = load_dataset_with_feedback("HuggingFaceH4/instruction-dataset", tokenizer=tokenizer, positive_feedback=True)



    # Simple verify
    active_features = {}
    for inputs in tqdm(dataset):
        tokenized_inputs = tokenizer(inputs["formatted_chat"], return_tensors="pt").to(model.device)
        target_activations = gather_residual_activations(model, target_layer=17, inputs=tokenized_inputs)
        sae_acts = sae.encode(target_activations)
        recon = sae.decode(sae_acts)
        values, inds = sae_acts.max(-1)
        for v, i in zip(values[0].tolist(), inds[0].tolist()):
            active_features[i] = v if i not in active_features else active_features[i] + v

    print("The most active features are:")
    for k in sorted(active_features, key=active_features.get, reverse=True):
        print(f"Feature {k} with total activation {active_features[k]:.4f}")



