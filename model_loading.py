import os
import getpass
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login, get_token
from sae_lens import SAE

# --- Configuration ---
MODEL_ID = "google/gemma-3-1b-it"
LOCAL_SAVE_DIRECTORY = "./gemma-3-model-local"
LOCAL_SAE_SAVE_DIRECTORY = "./gemma-scope-saes-local"

def check_login():
    """
    Checks if the user is logged in to Hugging Face.
    If not, requests the token interactively.
    """
    # Check if a token exists in the cache or environment
    if get_token() is not None:
        print("✅ Detected valid Hugging Face login credentials.")
        return

    # If not logged in, prompt the user
    print("\n🔒 Hugging Face Login Required (Gemma 3 is a gated model).")
    print("If you don't have a token, get one here: https://huggingface.co/settings/tokens")

    try:
        # getpass hides the input while typing for security
        user_token = getpass.getpass("👉 Paste your Access Token here: ")

        # Log in (this saves the token to ~/.cache/huggingface/token)
        login(token=user_token)
        print("✅ Login successful.")
    except Exception as e:
        print(f"❌ Login failed: {e}")
        exit(1)


def load_gemma_model_and_tokenizer(model_size: str = "1b", instructions_tuned: bool = True):
    # 1. Check if the local directory exists and contains the model
    local_save_dir = LOCAL_SAVE_DIRECTORY + "_" + model_size + ("_it" if instructions_tuned else "_pt")
    if os.path.isdir(local_save_dir) and \
            os.path.exists(os.path.join(local_save_dir, "config.json")):

        print(f"\n📂 Found local model in '{local_save_dir}'.")
        print("Loading from disk (Offline)...")

        try:
            tokenizer = AutoTokenizer.from_pretrained(local_save_dir)
            model = AutoModelForCausalLM.from_pretrained(
                local_save_dir,
                device_map="auto",
                torch_dtype="auto"
            )
            return tokenizer, model
        except Exception as e:
            print(f"❌ Error loading local model: {e}")
            print("Attempting to re-download...")

    # 2. If we reach here, we need to download.
    # FIRST: Ensure we are authenticated.
    check_login()

    print(f"\n⬇️ Downloading '{MODEL_ID}' from Hugging Face Hub...")
    print("(This may take a while depending on your internet connection)")

    try:
        # We don't need to pass `token=` explicitly here because check_login()
        # saved it to the machine's default location, which transformers uses automatically.
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            device_map="auto",
            torch_dtype="auto"
        )

        # 3. Save to local directory
        print(f"💾 Saving model to '{local_save_dir}'...")
        tokenizer.save_pretrained(local_save_dir)
        model.save_pretrained(local_save_dir)
        print("✅ Model saved successfully.")

        return tokenizer, model

    except Exception as e:
        print(f"❌ Failed to download model: {e}")
        print("Tip: Ensure you have accepted the license at https://huggingface.co/google/gemma-3-1b-it")
        return None, None

def load_gemma_scope_sae(model_size: str = "1b", instructions_tuned: bool = True, layer: int = 12, width: str = "262k"):
    local_save_dir = LOCAL_SAVE_DIRECTORY + "_" + model_size + ("_it" if instructions_tuned else "_pt")
    if os.path.isdir(local_save_dir):
        print(f"\n📂 Found local SAE in '{local_save_dir}'.")
        print("Loading from disk (Offline)...")

        try:
            sae = SAE.from_pretrained(local_save_dir)
            return sae
        except Exception as e:
            print(f"❌ Error loading local SAE: {e}")
            print("Attempting to re-download...")

    from sae_lens import SAE

    release = "gemma-scope-2-1b-it-att"
    sae_id = "layer_17_width_262k_l0_medium"
    print(f"\n⬇️ Downloading SAE '{sae_id}' from release '{release}'...")
    sae = SAE.from_pretrained(release, sae_id)
    print(f"💾 Saving SAE to '{local_save_dir}'...")
    sae.save_pretrained(local_save_dir)
    print("✅ SAE saved successfully.")
    return sae




if __name__ == "__main__":
    tokenizer, model = load_gemma_model_and_tokenizer()
    sae = load_gemma_scope_sae()
    if model:
        print("\n--- Model Ready ---")
        # Simple verify
        inputs = tokenizer("Hello Gemma!", return_tensors="pt").to(model.device)
        print("Test generation:", tokenizer.decode(model.generate(**inputs, max_new_tokens=10)[0]))
        if sae:
            print("SAE is also loaded and ready.")