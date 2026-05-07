import json
import sys
from pathlib import Path
import torch

sys.path.insert(0, "/home/coder/share/Evo-RL-quick")
sys.path.insert(0, "/home/coder/share/Evo-RL-quick/scripts/rlt_training")
from common import load_training_config
from lerobot.rlt.algorithm import RLTAlgorithm
from lerobot.rlt.policy import RLTPolicy
from lerobot.rlt.vla_adapter import DummyVLAAdapter
from lerobot.rlt.evaluator import evaluate_offline
from lerobot.rlt.offline_dataset import load_transition_cache

ROOT = Path("/home/coder/share/Evo-RL-quick")
CACHE = ROOT / "outputs/cache_critical_clean_20260507"
SWEEP = ROOT / "outputs/ac_sweep_critical_clean_20260507"
DEVICE = "cuda"

config = load_training_config(None)
val_buffer = load_transition_cache(str(CACHE), "val", capacity=config.replay.capacity)
print(f"loaded {len(val_buffer)} val transitions")

vla = DummyVLAAdapter(token_dim=config.rl_token.token_dim, action_dim=config.action_dim, num_tokens=64, horizon=config.vla_horizon)
policy = RLTPolicy(config, vla).to(DEVICE)
policy.freeze_vla(); policy.freeze_rl_token_encoder()
algorithm = RLTAlgorithm(policy, config)
algorithm.to(DEVICE)

results = {}
for name in ["b0.1", "b0.3", "b1.0"]:
    ckpt = SWEEP / name / "rl_checkpoint.pt"
    sd = torch.load(ckpt, map_location=DEVICE, weights_only=False)
    algorithm.policy.actor.load_state_dict(sd["actor_state_dict"])
    algorithm.critic.load_state_dict(sd["critic_state_dict"])
    algorithm.target_critic.load_state_dict(sd["target_critic_state_dict"])
    m = evaluate_offline(algorithm, val_buffer, config, num_batches=20)
    print(f"{name}: expert_mse={m.expert_action_mse:.4f} ref_mse={m.ref_action_mse:.4f} ref_dropped_mse={m.ref_dropped_mse:.4f} q_pol={m.mean_q_policy:.4f} q_exp={m.mean_q_expert:.4f} q_gap={m.q_gap:.4f} td={m.mean_critic_td_error:.4f}")
    results[name] = {
        "expert_mse": m.expert_action_mse, "ref_mse": m.ref_action_mse, "ref_dropped_mse": m.ref_dropped_mse,
        "q_policy": m.mean_q_policy, "q_expert": m.mean_q_expert, "q_gap": m.q_gap, "td_error": m.mean_critic_td_error,
    }

(SWEEP / "ref_dropped_mse_eval.json").write_text(json.dumps(results, indent=2))
print("written:", SWEEP / "ref_dropped_mse_eval.json")
