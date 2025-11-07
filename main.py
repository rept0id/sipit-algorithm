import torch
from transformers import GPT2Model, GPT2Tokenizer

### # # ###

CONST_MODEL_NAME = "gpt2"

CONST_DEFAULT_EPSILON = 1e-3
CONST_DEFAULT_MAX_LENGTH = 20

### # # ###


def get_hidden_states(input_text, model, tokenizer, layer=-1):
    """
    Extract hidden states H̃^(ℓ) ∈ ℝ^T×d for input_text (paper's observed states).

    Corresponds to: "Require: Observed layer-ℓ states H̃^(ℓ) ∈ ℝ^T×d"
    """
    inputs = tokenizer(input_text, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    return outputs.hidden_states[layer]  # Default: last layer (ℓ = L)


def sip_it_reconstruct(
    hidden_states,
    model,
    tokenizer,
    max_length=CONST_DEFAULT_MAX_LENGTH,
    epsilon=CONST_DEFAULT_EPSILON,
):
    """
    Exact implementation of Algorithm 1 (Page 6).

    Args:
     - hidden_states: observed hidden states H̃^(ℓ) ∈ ℝ^T×d (Algorithm 1, Require)
     - model : model reference
     - tokenizer : tokenizer reference
     - max_length: maximum sequence length T (Algorithm 1, for t=1 to T)
     - epsilon: tolerance ε ≥ 0 (Algorithm 1, Require)

    Returns:
        Reconstructed sequence s̃ = ⟨s̃₁, ..., s̃_T⟩ (Algorithm 1, Ensure).
    """

    # Line 1: Initialize empty sequence s̃ ← ⟨⟩
    reconstructed_ids = []
    prefix = []  # π = reconstructed prefix (Algorithm 1, implicit in F(v; π, t))

    # Get original sequence length from hidden_states (T)
    original_seq_len = hidden_states.shape[1]

    ### # # ###

    # Line 2: for t = 1 to T
    for t in range(min(max_length, original_seq_len)):
        # Line 3: 𝒞 ← ∅ (set of tested candidates)
        tested_candidates = set()

        # Line 4: for j = 1 to |𝒱| (loop over vocabulary)
        best_distance = float("inf")
        best_token_id = None

        ### # # ###

        for token_id in range(len(tokenizer)):
            if token_id in tested_candidates:
                continue  # Skip duplicates (POLICY avoids repeats)
            tested_candidates.add(token_id)

            # Line 5: v_j ← POLICY(𝒱, 𝒞, s̃, ℓ)
            # Here, POLICY = exhaustive search (no gradients for simplicity)
            candidate_prefix = prefix + [token_id]

            # Reconstruct input tensor with attention_mask (critical for GPT-2)
            candidate_input = {
                "input_ids": torch.tensor([candidate_prefix]).to("cuda"),
                "attention_mask": torch.ones(1, len(candidate_prefix)).to("cuda"),
            }

            # Compute candidate hidden state h_t(π ⊕ v_j) (Line 5, F(v; π, t))
            with torch.no_grad():
                candidate_outputs = model(**candidate_input, output_hidden_states=True)
                # Ensure t is within bounds for the candidate sequence
                if t < candidate_outputs.hidden_states[-1].shape[1]:
                    candidate_hidden = candidate_outputs.hidden_states[-1][
                        0, t, :
                    ]  # h_t(π ⊕ v_j)
                else:
                    continue  # Skip if t exceeds candidate sequence length

            # Line 6: Check if h̃_t ∈ 𝒜_{π,t}(v_j; ε)
            # 𝒜_{π,t}(v_j; ε) = {h | ||h - h_t(π ⊕ v_j)||₂ ≤ ε}
            distance = torch.norm(hidden_states[0, t, :] - candidate_hidden, p=2).item()

            # Update best pick
            if distance < best_distance:
                best_distance = distance
                best_token_id = token_id

        # Line 7–8: s̃ ← s̃ ⊕ v_j (append token) and break (move to next t)
        if best_token_id is not None:
            reconstructed_ids.append(best_token_id)
            prefix = reconstructed_ids.copy()  # Update π for next iteration

            print(f"Token {best_token_id} ('{tokenizer.decode([best_token_id])}')")
        else:
            print(f"Warning: No token found at position {t} with ε={epsilon}.")

            break  # Algorithm 1, Line 9–11 (failure case)

        ### # # ###

        # Early exit if full sequence is reconstructed (not in algorithm, but practical)
        if len(reconstructed_ids) == original_seq_len:
            break

    ### # # ###

    # Line 14: return s̃
    return reconstructed_ids


def example():
    LOCAL_CONST_PROMPT = "The quick brown fox"

    ### # # ###

    # Load GPT-2 (decoder-only Transformer, satisfies paper's assumptions)
    #
    model = GPT2Model.from_pretrained(
        CONST_MODEL_NAME
    ).cuda()  # Line: Model setup (not in algorithm, but required)
    #
    tokenizer = GPT2Tokenizer.from_pretrained(CONST_MODEL_NAME)

    # Disable gradients (paper assumes deterministic forward passes)
    #
    model.eval()
    #
    torch.no_grad()

    ### # # ###

    # Example usage (paper's Figure 1: prompt → latent space → SIP-It inversion)
    #
    hidden_states = get_hidden_states(
        LOCAL_CONST_PROMPT, model, tokenizer
    )  # Get H̃^(ℓ)(s)
    print(
        "Original hidden states shape:", hidden_states.shape
    )  # [1, seq_len, hidden_size]
    #
    reconstructed_ids = sip_it_reconstruct(
        hidden_states, model, tokenizer, max_length=20, epsilon=CONST_DEFAULT_EPSILON
    )  # Algorithm 1
    #
    reconstructed_text = tokenizer.decode(reconstructed_ids)

    ### # # ###

    print(f"Original: {LOCAL_CONST_PROMPT}")
    print(f"Reconstructed: {reconstructed_text}")


### # # ###


def main():
    print("Running sipit-algorithm example, please be patient through all the steps...")

    example()


if __name__ == "__main__":
    main()
