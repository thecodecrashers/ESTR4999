import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm

from lineEnv import lineEnv
from maml_neurwin import Actor  # We only need the Actor class from your training code

# ----------------------------
# 1) ROLL-OUT HELPER FUNCTIONS
# ----------------------------

Temperature = 1.0

def rollout_one_episode(actor, env, lam, gamma, device, max_steps=200):
    """
    Roll out one episode from the *current* env.state or from env.reset()
    and compute:
      sum_{t=0..T-1} [ G_t * log_prob(a_t) ]
    where G_t is the discounted return from step t to the end.
    """
    # Typically we do env.reset() here if we want a fresh start.
    # But if you want to fix env.state externally, do env.state = s before calling.
    # We'll do it here to ensure we start a fresh episode:
    env.reset()

    log_probs = []
    rewards   = []

    for step_i in range(max_steps):
        s_t = torch.tensor([env.state], dtype=torch.float32, device=device).unsqueeze(0)
        # Forward pass (no torch.no_grad(), we WANT gradients)
        logit = actor(
            s_t,
            torch.tensor([env.p, env.q, float(env.OptX)], 
                         dtype=torch.float32, device=device).unsqueeze(0)
        )
        # Adjust by lam, then get p(a=1)
        adj_logit = (logit - lam) / Temperature
        p1 = torch.sigmoid(adj_logit)          # shape (1,1)

        probs = torch.cat([1 - p1, p1], dim=-1)  # shape (1,2)
        dist = torch.distributions.Categorical(probs=probs)
        action = dist.sample()                   # 0 or 1
        log_prob = dist.log_prob(action)

        # Step environment
        next_state, reward, done, _ = env.step(action.item())
        if action.item() == 1:
            reward -= lam

        log_probs.append(log_prob)
        rewards.append(reward)

        if done:
            break

    T = len(rewards)
    if T == 0:
        return torch.tensor(0.0, device=device)

    rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)
    log_probs_t = torch.stack(log_probs)

    # Compute discounted returns
    returns_t = torch.zeros(T, dtype=torch.float32, device=device)
    running_return = 0.0
    for t in reversed(range(T)):
        running_return = rewards_t[t] + gamma * running_return
        returns_t[t] = running_return

    # sum_{t=0..T-1}[ G_t * log_prob(a_t) ]
    total_obj = torch.sum(returns_t * log_probs_t)
    return total_obj


def rollout_one_episode_from_state(actor, env, start_state, lam, gamma, device, max_steps=200):
    """
    Same as rollout_one_episode, except we explicitly set env.state = start_state
    before rolling out.
    """
    env.reset()
    env.state = start_state
    return rollout_one_episode(actor, env, lam, gamma, device, max_steps)


def reinforce_loss_over_all_states(actor, env, lam, gamma, device, N, max_steps=200):
    """
    For each state in [0..N-1], do one rollout (i.e. reset + set env.state = s),
    sum up sum_{t}[ G_t * logπ(a_t|s_t)] across states.
    """
    total_obj = torch.tensor(0.0, device=device)
    for s in range(N):
        # rollout from that state
        ep_obj = rollout_one_episode_from_state(
            actor, env, s, lam, gamma, device, max_steps
        )
        total_obj += ep_obj
    return total_obj


def clone_model(model: nn.Module):
    """Simple copy of a PyTorch module."""
    import copy
    return copy.deepcopy(model)


# ----------------------------
# 2) ADAPTATION FOR A SINGLE ARM
# ----------------------------

def adapt_single_arm(
    meta_actor: nn.Module,
    env: lineEnv,
    device,
    lam_min=0.0,
    lam_max=2.0,
    adaptation_steps=200,
    gamma=0.99,
    inner_lr=1e-4,
    K=30,
    max_rollout_len=200
):
    """
    Given a trained meta_actor, adapt it to a single arm (single environment) by
    performing REINFORCE updates.
    - We do K rounds; in each round:
      1) Sample lam in [lam_min, lam_max].
      2) Optionally compare meta_actor's output to see if lam is in [0,2].
      3) For 'adaptation_steps' gradient steps, compute the REINFORCE objective
         by rolling out from all states [0..N-1], then do gradient ascent.

    Return the adapted actor.
    """
    actor_fast = clone_model(meta_actor).to(device)
    fast_actor_optim = optim.Adam(actor_fast.parameters(), lr=inner_lr)

    N = env.N  # number of states in lineEnv, e.g. 100

    for _ in range(K):
        # Sample lam
        lam = random.uniform(lam_min, lam_max)

        # Optionally compare meta_actor's output with lam
        # This logic was in your original code, but be sure it's what you want.
        s_int = np.random.randint(1, N + 1)  # pick a random state in [1..N]
        s_tensor = torch.tensor([s_int], dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            value = meta_actor(
                s_tensor,
                torch.tensor([env.p, env.q, float(env.OptX)],
                             dtype=torch.float32, device=device).unsqueeze(0)
            )
        if isinstance(value, torch.Tensor):
            value = value.item()
        if (0 <= value <= 2):
            lam = value  # override lam

        # Do 'adaptation_steps' gradient updates for this lam
        for step_i in range(adaptation_steps):
            fast_actor_optim.zero_grad()
            # big_objective is sum of REINFORCE returns across all states
            big_objective = reinforce_loss_over_all_states(
                actor_fast, env, lam, gamma, device, N, max_rollout_len
            )
            adapt_loss = -big_objective  # gradient ascent => negative objective
            adapt_loss.backward()
            fast_actor_optim.step()

    return actor_fast


# ----------------------------
# 3) TESTING MULTI-ARMS w/ TOP-K
# ----------------------------

def test_multi_arms_top_k(actors_list, envs_list, device,
                          k=3,
                          total_steps=10000,
                          warmup_steps=2000,
                          report_interval=200):
    """
    For each timestep:
      1) Evaluate each arm's logit on the current state => pick top-k arms => action=1, else 0
      2) Step each environment with its chosen action => get reward
      3) After warmup_steps, measure average reward.
    """
    num_arms = len(envs_list)
    states = [env.reset() for env in envs_list]

    total_reward_after_warmup = 0.0
    steps_after_warmup = 0
    report_records = []

    for step_i in range(1, total_steps + 1):
        # 1) compute logit for each arm
        logits = []
        for i in range(num_arms):
            s_t = torch.tensor([states[i]], dtype=torch.float32, device=device)
            env_i = envs_list[i]
            env_embed_3d = torch.tensor([env_i.p, env_i.q, float(env_i.OptX)],
                                        dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                logit = actors_list[i](s_t, env_embed_3d)
                logits.append(logit.item())

        # 2) pick top-k arms => action=1, else action=0
        chosen_indices = np.argsort(logits)[-k:]
        chosen_set = set(chosen_indices)

        # 3) step each environment
        step_reward_sum = 0.0
        for i in range(num_arms):
            action = 1 if i in chosen_set else 0
            next_state, reward, done, _ = envs_list[i].step(action)
            step_reward_sum += reward
            states[i] = next_state
            if done:
                states[i] = envs_list[i].reset()

        # 4) accumulate reward if beyond warmup
        if step_i > warmup_steps:
            total_reward_after_warmup += step_reward_sum
            steps_after_warmup += 1

            if step_i % report_interval == 0:
                avg_r = total_reward_after_warmup / steps_after_warmup
                report_records.append((step_i, avg_r))

    return report_records


def plot_logits_across_states(actors_list, envs_list, device):
    """
    Plot each environment's logit across states [1..100].
    """
    num_arms = len(envs_list)
    states_range = np.arange(1, 101)
    
    plt.figure(figsize=(10, 6))

    for i in range(num_arms):
        env_i = envs_list[i]
        actor_i = actors_list[i]
        
        env_embed_3d = torch.tensor([env_i.p, env_i.q, float(env_i.OptX)],
                                    dtype=torch.float32, device=device).unsqueeze(0)
        logits = []
        for state_val in states_range:
            s_t = torch.tensor([state_val], dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                logit = actor_i(s_t, env_embed_3d).item()
                logits.append(logit)
        
        plt.plot(states_range, logits, label=f"Arm {i} (p={env_i.p:.2f}, q={env_i.q:.2f})")

    plt.xlabel("State (1-100)")
    plt.ylabel("Logit")
    plt.title("Logits for Each Environment Across States")
    plt.legend()

    out_dir = "maml_neurwin_lin_out"
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, "logits_across_states.png"))
    plt.show()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 1) Load trained meta-Actor from your MAML code
    meta_actor_path = "maml_neurwin/meta_actor.pth"
    meta_actor = Actor(state_dim=1).to(device)
    meta_actor.load_state_dict(torch.load(meta_actor_path, map_location=device))
    meta_actor.eval()

    # 2) Create multiple single-arm environments
    num_arms = 10
    N = 100
    OptX = 99
    p_vals = np.linspace(start=0.2, stop=0.8, num=num_arms)
    q_vals = np.linspace(start=0.8, stop=0.2, num=num_arms)

    arms_data = []
    for i in range(num_arms):
        p_val = p_vals[i]
        q_val = q_vals[i]
        seed_val = random.randint(0, 99999)
        arms_data.append((p_val, q_val, seed_val))

    # 3) Adapt each arm (ENV) from the meta-actor
    adaptation_steps = 20   # number of gradient steps per lam
    adapt_inner_lr = 1e-3
    K = 1  # how many lam draws we do => K * adaptation_steps total updates

    arm_actors = []
    test_envs = []
    for i, (p_val, q_val, seed_val) in enumerate(arms_data):
        env_i = lineEnv(seed=seed_val, N=N, OptX=OptX, p=p_val, q=q_val)
        fast_actor = adapt_single_arm(
            meta_actor=meta_actor,
            env=env_i,
            device=device,
            lam_min=0.0,
            lam_max=2.0,
            adaptation_steps=adaptation_steps,
            gamma=0.99,
            inner_lr=adapt_inner_lr,
            K=K,
            max_rollout_len=50  # how many steps in each environment rollout
        )
        arm_actors.append(fast_actor)
        # We'll also store a fresh env for testing
        test_envs.append(lineEnv(seed=seed_val, N=N, OptX=OptX, p=p_val, q=q_val))
        print(f"[Arm {i}] p={p_val:.3f}, q={q_val:.3f}, seed={seed_val} => Adapted.")

    # 4) (Optional) Plot the resulting logits across states
    plot_logits_across_states(arm_actors, test_envs, device)

    # 5) Multi-arm top-k test
    total_steps_test = 10000
    warmup_steps = 2000
    report_interval = 200
    top_k = 3

    results = test_multi_arms_top_k(
        actors_list=arm_actors,
        envs_list=test_envs,
        device=device,
        k=top_k,
        total_steps=total_steps_test,
        warmup_steps=warmup_steps,
        report_interval=report_interval
    )

    # 6) Plot the average reward over time
    steps_plt = [r[0] for r in results]
    avgR_plt = [r[1] for r in results]

    plt.figure(figsize=(7,5))
    plt.plot(steps_plt, avgR_plt, marker='o', label=f"Top-{top_k} selection average reward")
    plt.xlabel("Global Step")
    plt.ylabel("Average Reward")
    plt.title(f"Test MAML-Neurwin Multi-Arms (Top-k = {top_k})")
    plt.legend()

    out_dir = "maml_neurwin_lin_out"
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, "test_result_top_k.png"))
    plt.show()

    if len(avgR_plt) > 0:
        print("[Info] Final reported average reward =", avgR_plt[-1])
    else:
        print("[Info] No reported data.")


if __name__ == "__main__":
    main()

