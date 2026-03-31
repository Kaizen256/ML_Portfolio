import random
import textwrap
from dataclasses import dataclass

# =========================
# RL Bucket Quiz (No Options)
# =========================
# How it works:
# - You see an algorithm name.
# - You type the BUCKET (no multiple-choice).
# - Press Enter to reveal the answer + definitions.
#
# Buckets to type (case-insensitive; synonyms accepted):
#   A: value-based (Q only)
#   B: policy-based / on-policy policy gradient
#   C: actor-critic off-policy continuous control
#   D: model-based planning
#   E: sequence modeling / offline-ish
#
# Tip: You can type just A/B/C/D/E or the bucket name.
# =========================

WRAP = 92

def wrap(s: str) -> str:
    return "\n".join(textwrap.wrap(s.strip(), width=WRAP))

@dataclass
class Card:
    name: str
    bucket_code: str
    bucket_name: str
    one_liner: str
    explain: str

CARDS = [
    Card(
        name="DQN",
        bucket_code="A",
        bucket_name="Value-based (Q only) | mostly discrete actions",
        one_liner="Learns a Q-score for each discrete action and picks the biggest; uses replay + target network so training doesn’t implode.",
        explain=(
            "DQN (Deep Q-Network) turns Q-learning into deep learning. The network predicts how good each action is in a state. "
            "Training is unstable because targets depend on the network itself, so DQN uses a target network to compute targets "
            "and a replay buffer to avoid learning from highly correlated consecutive experience."
        ),
    ),
    Card(
        name="PPO",
        bucket_code="B",
        bucket_name="Policy-based (constrained policy updates) | on-policy | stable baseline",
        one_liner="Learns a policy directly but limits how much it can change per update; stable and scalable but needs lots of fresh data.",
        explain=(
            "PPO (Proximal Policy Optimization) is a policy-gradient method that improves stability by preventing the new policy from moving too far from the old policy each update. "
            "That makes it very reliable and scalable, but since it’s on-policy it needs lots of new rollouts and is less sample efficient."
        ),
    ),
    Card(
        name="SAC",
        bucket_code="C",
        bucket_name="Actor-Critic off-policy continuous control | stochastic actor + entropy",
        one_liner="Off-policy actor-critic for continuous actions that intentionally stays random via entropy; robust and sample efficient.",
        explain=(
            "SAC (Soft Actor-Critic) is an off-policy actor-critic method that adds an entropy objective so the policy stays exploratory. "
            "That tends to make it more robust than deterministic methods, and replay makes it sample efficient. "
            "It’s a strong default for continuous control when environment steps are expensive."
        ),
    ),
    Card(
        name="DDPG",
        bucket_code="C",
        bucket_name="Actor-Critic off-policy continuous control | deterministic actor | can be finicky",
        one_liner="Continuous control actor-critic with a deterministic actor; off-policy with replay; can be unstable.",
        explain=(
            "DDPG (Deep Deterministic Policy Gradient) is basically 'DQN for continuous actions' using an actor-critic setup. The critic learns Q(s,a), "
            "and the actor is trained to choose actions that maximize the critic’s Q. It’s off-policy with replay, "
            "but it can be unstable and sensitive to hyperparameters."
        ),
    ),
    Card(
        name="TD3",
        bucket_code="C",
        bucket_name="Actor-Critic off-policy continuous control | DDPG fixed (twin critics, delay, smoothing)",
        one_liner="DDPG but fixed: two critics, delayed actor updates, and target smoothing to stop overestimation and jitter.",
        explain=(
            "TD3 (Twin Delayed Deep Deterministic Policy Gradient) is DDPG made stable. It tackles Q overestimation with two critics and clipped targets, updates the actor less frequently, "
            "and smooths target actions. The result is much more reliable learning in continuous control."
        ),
    ),
    Card(
        name="MuZero",
        bucket_code="D",
        bucket_name="Model-based planning | latent dynamics + search (MCTS-style planning)",
        one_liner="Learns its own internal model of the environment in latent space and uses search/planning to pick actions.",
        explain=(
            "MuZero learns a latent dynamics model and uses planning to choose actions without needing the true environment rules. "
            "It learns only what’s needed for good decisions: a representation of state, how it evolves under actions, and value/policy predictions "
            "used during search."
        ),
    ),
    Card(
        name="Decision Transformer",
        bucket_code="E",
        bucket_name="Sequence modeling / Offline-ish | 'RL as conditional sequence prediction'",
        one_liner="Doesn’t learn Q or policy gradients; it predicts actions like language modeling conditioned on desired return.",
        explain=(
            "Decision Transformer reframes RL as sequence modeling. Instead of learning a value function or using policy gradients, "
            "it learns to predict actions conditioned on a target return and past trajectory. It’s especially relevant for offline RL "
            "when you have lots of logged trajectories."
        ),
    ),
]

# Acceptable user inputs -> bucket code
BUCKET_ALIASES = {
    "A": "A",
    "VALUE": "A",
    "VALUE-BASED": "A",
    "VALUEBASED": "A",
    "Q": "A",
    "Q-LEARNING": "A",
    "QLEARNING": "A",

    "B": "B",
    "POLICY": "B",
    "POLICY-BASED": "B",
    "POLICYBASED": "B",
    "POLICY GRADIENT": "B",
    "POLICYGRADIENT": "B",
    "ON-POLICY": "B",
    "ONPOLICY": "B",
    "PPO": "B",  # if you accidentally type the algo name

    "C": "C",
    "ACTOR": "C",
    "ACTOR-CRITIC": "C",
    "ACTORCRITIC": "C",
    "OFF-POLICY": "C",
    "OFFPOLICY": "C",
    "CONTINUOUS": "C",

    "D": "D",
    "MODEL": "D",
    "MODEL-BASED": "D",
    "MODELBASED": "D",
    "PLANNING": "D",
    "SEARCH": "D",

    "E": "E",
    "SEQUENCE": "E",
    "SEQUENCE MODELING": "E",
    "SEQUENCEMODELING": "E",
    "TRANSFORMER": "E",
    "OFFLINE": "E",
}

def normalize_bucket_input(s: str) -> str | None:
    s = s.strip().upper()
    if not s:
        return None
    # normalize spaces and punctuation lightly
    s = s.replace("_", " ").replace("/", " ").replace("\\", " ")
    s = " ".join(s.split())
    # direct
    if s in BUCKET_ALIASES:
        return BUCKET_ALIASES[s]
    # try compact (remove spaces/hyphens)
    compact = s.replace(" ", "").replace("-", "")
    if compact in BUCKET_ALIASES:
        return BUCKET_ALIASES[compact]
    return None

def print_header():
    print("\n" + "=" * WRAP)
    print("RL BUCKET QUIZ (NO OPTIONS)")
    print("=" * WRAP)
    print("Type the bucket (A/B/C/D/E) or a name like 'value', 'actor-critic', 'model-based', 'sequence'.")
    print("Commands:  'quit' to stop | 'stats' to view stats | 'help' to reprint this")
    print("-" * WRAP)

def print_help():
    print("\nBuckets you can type (no multiple-choice will be shown during quiz):")
    print("  A = Value-based (Q only) | mostly discrete")
    print("  B = Policy-based / constrained policy updates | on-policy")
    print("  C = Actor-Critic off-policy continuous control (DDPG/TD3/SAC family)")
    print("  D = Model-based planning (learn model + plan/search)")
    print("  E = Sequence modeling / offline-ish (Decision Transformer style)\n")

def show_card_answer(card: Card, user_bucket: str | None):
    correct = (user_bucket == card.bucket_code)
    print("\n" + "-" * WRAP)
    if user_bucket is None:
        print("Your answer: (blank)")
    else:
        print(f"Your answer: {user_bucket}")
    print(f"Correct:     {card.bucket_code}  |  {card.bucket_name}")
    print("Result:      " + ("✅ correct" if correct else "❌ incorrect"))
    print("-" * WRAP)
    print("12-second one-liner:")
    print(wrap(card.one_liner))
    print("\nExplain out loud:")
    print(wrap(card.explain))
    print("-" * WRAP)
    return correct

def show_stats(stats):
    # stats: dict bucket_code -> [correct, total]
    print("\n" + "=" * WRAP)
    print("STATS")
    print("=" * WRAP)
    total_correct = 0
    total = 0
    for code in ["A", "B", "C", "D", "E"]:
        c, t = stats.get(code, [0, 0])
        total_correct += c
        total += t
        pct = (100.0 * c / t) if t else 0.0
        print(f"{code}: {c:>3}/{t:<3}  ({pct:>5.1f}%)")
    overall = (100.0 * total_correct / total) if total else 0.0
    print("-" * WRAP)
    print(f"Overall: {total_correct}/{total}  ({overall:.1f}%)")
    print("=" * WRAP)

def quiz_loop(seed: int | None = None, shuffle_each_round: bool = True):
    if seed is not None:
        random.seed(seed)

    stats = {code: [0, 0] for code in ["A", "B", "C", "D", "E"]}

    print_header()
    print_help()

    deck = CARDS[:]
    i = 0

    while True:
        if i >= len(deck):
            i = 0
            if shuffle_each_round:
                random.shuffle(deck)
            print("\n--- new round ---\n")

        card = deck[i]
        i += 1

        print(f"\nAlgorithm: {card.name}")
        ans = input("Bucket? ").strip()

        if ans.lower() in {"quit", "exit", "q"}:
            print("\nStopping. Humans love closure.\n")
            show_stats(stats)
            break

        if ans.lower() in {"help", "h", "?"}:
            print_help()
            continue

        if ans.lower() in {"stats", "s"}:
            show_stats(stats)
            continue

        user_bucket = normalize_bucket_input(ans)

        # Wait for "I'm ready" moment: press Enter to reveal
        input("Press Enter to reveal... ")

        correct = show_card_answer(card, user_bucket)

        # Update stats (count only if user typed something recognizable)
        if user_bucket is not None:
            stats[user_bucket][1] += 1
        else:
            # If blank/unknown, count as attempt under a special bucket? We'll count as overall miss by not adding.
            pass

        if user_bucket is not None:
            if correct:
                stats[user_bucket][0] += 1

        # Small pause prompt so it doesn't fly past
        input("Press Enter for next... ")

if __name__ == "__main__":
    # seed=None gives true randomness; set an int for repeatable order
    quiz_loop(seed=None, shuffle_each_round=True)