import os
import sqlite3
import logging
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import (
    PPOTrainer,
    PPOConfig,
    AutoModelForCausalLMWithValueHead,
    create_reference_model
)

###############################################################################
# LOGGING CONFIGURATION
###############################################################################
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

###############################################################################
# CONFIG CLASS (Reads HF token)
###############################################################################
class Config:
    def __init__(self, hf_token_file="hf_token.txt"):
        if not os.path.exists(hf_token_file):
            raise FileNotFoundError(
                f"HF token file '{hf_token_file}' not found. "
                "Please create it and place your token inside."
            )
        with open(hf_token_file, "r") as f:
            self.hf_token = f.read().strip()

###############################################################################
# DATASET LOADER FROM DB
###############################################################################
class FeedbackDataset(Dataset):
    def __init__(self, db_path="feedback.db"):
        self.samples = []
        if not os.path.exists(db_path):
            logger.warning(f"Database file '{db_path}' does not exist.")
            return

        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        c.execute("SELECT question, edited_response, score FROM feedback")
        rows = c.fetchall()
        conn.close()

        for question, edited_answer, score in rows:
            score = float(score) if score is not None else 0.0
            if edited_answer:
                self.samples.append((question, edited_answer, score))

        logger.info(f"Loaded {len(self.samples)} samples from {db_path}.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

###############################################################################
# REWARD FUNCTION
###############################################################################
def compute_reward(generated_text, reference_text, user_score):
    """
    A dummy reward that combines a user score with a token overlap factor.
    """
    gen_tokens = set(generated_text.split())
    ref_tokens = set(reference_text.split())
    overlap = len(gen_tokens.intersection(ref_tokens)) / (len(ref_tokens) + 1e-9)

    alpha = 0.5
    reward = user_score + alpha * overlap
    # Clip the reward so it doesn't explode or go negative
    return min(reward, 2.0)

###############################################################################
# MAIN TRAINING LOOP
###############################################################################
def main():
    # --------------------------------------------------------------------------
    # 1) Load dataset
    # --------------------------------------------------------------------------
    dataset = FeedbackDataset()
    if len(dataset) == 0:
        logger.info("No data found. Exiting.")
        return

    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # --------------------------------------------------------------------------
    # 2) Load your HF token
    # --------------------------------------------------------------------------
    try:
        config = Config("hf_token.txt")
        hf_auth = config.hf_token
        logger.info("HuggingFace token loaded.")
    except FileNotFoundError as e:
        logger.error(str(e))
        return

    # --------------------------------------------------------------------------
    # 3) Initialize model + tokenizer
    # --------------------------------------------------------------------------
    model_id = "meta-llama/Llama-3.1-8B-Instruct"
    logger.info(f"Using model: {model_id}")

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=hf_auth)
    tokenizer.pad_token = tokenizer.eos_token  # ensure pad token is set

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float32
    )

    # Load the base CausalLM model
    logger.info("Loading base model...")
    
    
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        device_map="auto",
        use_auth_token=hf_auth
        # quantization_config=bnb_config,  # If you want to enable 4-bit
    )

    # Wrap base model with a value head
    model = AutoModelForCausalLMWithValueHead(base_model)

    # Create a reference model for KL-divergence calculation
    ref_model = create_reference_model(model)  # <--- from the snippet

    # --------------------------------------------------------------------------
    # 4) PPO Trainer Configuration
    # --------------------------------------------------------------------------
    # We use a small batch size=1 for illustration. Increase if you have more GPU memory.
    ppo_config = PPOConfig(
        batch_size=1,
        mini_batch_size=1,
        gradient_accumulation_steps=1,
        # Optional: define some PPO parameters if you like:
        # kl_coef=0.1, cliprange=0.2, vf_coef=0.1, etc.
        num_ppo_epochs=1  # number of epochs to train over the dataset
    )

    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=model,
        ref_model=ref_model,  # <--- use the newly created reference model
        tokenizer=tokenizer
    )

    # --------------------------------------------------------------------------
    # 5) (Optional) Example single-step to show respond_to_batch usage
    # --------------------------------------------------------------------------
    # This replicates the snippet's “single query to response” example:
    # encode single query
    query_txt = "This morning I went to the "
    query_tensor = tokenizer.encode(query_txt, return_tensors="pt").to(model.device)

    # generate response using PPOTrainer's built-in generation
    response_tensor = ppo_trainer.generate(query_tensor)

    # define dummy reward
    reward = [torch.tensor(1.0, device=model.device)]

    # run PPO update step
    train_stats = ppo_trainer.step([query_tensor[0]], [response_tensor[0]], reward)
    logger.info(f"[Illustration only] Single-step PPO stats: {single_stats}")

    # --------------------------------------------------------------------------
    # 6) Full multi-epoch training loop
    # --------------------------------------------------------------------------
    generation_kwargs = {
        "max_new_tokens": 50,     # or whatever you like
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
    }
    
    # After model + tokenizer init
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    logger.info("Starting PPO training...")
    for epoch in range(ppo_config.num_ppo_epochs):
        logger.info(f"Epoch {epoch} starting...")
        for step, (question, edited_answer, score) in enumerate(dataloader):
            # Prepare the prompt
            prompt = f"Human: {question}\nAssistant:"
            query_tensor = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

            # Generate a response (using respond_to_batch)
            rresponse_tensor = ppo_trainer.generate(query_tensor)

            # Decode to text
            generated_text = tokenizer.decode(
                response_tensor[0][query_tensor.shape[-1]:],
                skip_special_tokens=True
            )

            # Compute reward
            reward_value = compute_reward(generated_text, edited_answer, score)
            reward_tensor = torch.tensor([reward_value], dtype=torch.float32).to(model.device)

            # PPO step
            stats = ppo_trainer.step(
                queries=[query_tensor[0]],
                responses=[response_tensor[0]],
                rewards=reward_tensor
            )

            # Log stats each step
            logger.info(
                f"Step {step} | Reward={reward_value:.4f} | "
                f"kl_divergence={stats.get('kl_divergence', 'n/a'):.4f}, "
                f"policy_entropy={stats.get('policy_entropy', 'n/a'):.4f}"
            )

        logger.info(f"Epoch {epoch} complete.")

    # --------------------------------------------------------------------------
    # 7) Save the final model
    # --------------------------------------------------------------------------
    output_dir = "./rl_model"
    os.makedirs(output_dir, exist_ok=True)
    ppo_trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"RLHF model saved to {output_dir}.")

if __name__ == "__main__":
    main()
