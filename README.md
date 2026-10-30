# Snake AI Lab

**Snake AI Lab** is a Snake game with a Deep Q-Network (DQN) agent that learns to play through reinforcement learning.

This repository contains the game environment, CPU-based DQN training, evaluation, and a Pygame interface for manual play and AI Auto Play.

## What it does

- Trains a DQN agent to collect food and avoid collisions.
- Runs training in a headless environment, separate from the Pygame interface.
- Saves the trained model as a checkpoint.
- Evaluates the trained model over multiple episodes.
- Compares DQN performance with a random-action baseline.
- Lets you play manually or watch the trained agent in Pygame.

## Quick start (Windows)

### Requirements

- Windows 10 or 11
- **Python 3.12 recommended** (Python 3.11+ may work if compatible PyTorch wheels are available)
- Internet connection for the first-time dependency installation
- VS Code is optional

### 1. Download the repository

Clone it:

```powershell
git clone https://github.com/<your-username>/snake-ai-lab.git
cd snake-ai-lab
```

Or use GitHub's **Code → Download ZIP**, extract it, and open the `snake-ai-lab` folder in PowerShell.

### 2. Run the application

```powershell
.\run.bat
```

On first launch, the script creates a `.venv` virtual environment, installs CPU-only PyTorch, Pygame and test dependencies, and runs the tests. Then it launches the app.

On later launches, run `.\run.bat` again. It reuses the installed environment.

> Run commands from the repository root—the folder containing `run.bat`. You do not need to activate `.venv` manually.

## Recommended workflow

### Option A — Train, evaluate, then open the app (one command)

```powershell
.\train-evaluate-demo.bat
```

This script:
1. Sets up the Python environment if needed.
2. Trains the DQN for 500 episodes on CPU.
3. Evaluates the trained model over 100 episodes.
4. Compares it with a random-action baseline using the same episode seeds.
5. Opens the Pygame application, where **AI Auto Play** is ready to select.

The first setup and training run may take several minutes. Keep the terminal open until the application starts.

### Option B — Use the application menu

Run `.\run.bat`, then choose from the menu:

| Mode | What it does |
|---|---|
| **Manual Play** | Play Snake using the keyboard. |
| **AI Auto Play** | Watch the saved DQN checkpoint play. |
| **Train DQN (500 episodes)** | Train or resume training inside the app. |
| **Evaluate DQN (100 episodes)** | Evaluate the saved checkpoint inside the app. |

A trained checkpoint is included at `models/checkpoints/dqn_snake.pt`, so **AI Auto Play** and **Evaluate DQN** are ready immediately. Select **Train DQN** to train a new model.

## Controls

- **Arrow keys / W A S D** — move in Manual Play
- **P / Space** — pause or resume
- **R / Enter** — restart after Game Over
- **Esc** — return to menu
- **Speed slider** — adjust visual playback speed

In AI Auto Play, the speed slider changes how quickly the game is displayed. It does not change the trained model or its policy.

## How the project works

```text
Snake environment
      |
      v
19-feature state observation
      |
      v
DQN agent chooses an action
      |
      v
Environment returns reward + next state
      |
      v
Experience replay + neural network update
      |
      v
Saved model checkpoint
      |
      +--> Pygame AI Auto Play
      |
      +--> DQN evaluation
      |
      +--> DQN vs random baseline comparison
```

### DQN design

- **State:** 19 `float32` features describing danger, direction, food and nearby body occupancy.
- **Actions:** straight, turn left, turn right.
- **Network:** 19 → 128 → 128 → 3, with ReLU hidden activations.
- **Algorithm:** DQN with experience replay, target network, epsilon-greedy exploration, Smooth L1 loss and gradient clipping.
- **Rewards:** +10 for food, -0.01 for a normal step, -10 for collision or starvation.
- **Training device:** CPU.

Training and inference are separated from rendering, so the model can be trained without drawing every game frame.

## Evaluation

After training, compare the DQN against a random-action policy:

```powershell
.\.venv\Scripts\python.exe evaluate.py --compare --episodes 100
```

Evaluate only the DQN:

```powershell
.\.venv\Scripts\python.exe evaluate.py --episodes 100
```

Evaluate only the random baseline:

```powershell
.\.venv\Scripts\python.exe evaluate.py --random --episodes 100
```

Evaluation reports the number of episodes, average score, best score, average steps and episode scores. The comparison uses the same episode seeds for DQN and random baseline.

### Included checkpoint results

One recorded evaluation of the project reported:

| Metric | DQN | Random baseline |
|---|---:|---:|
| Episodes | 100 | 100 |
| Average score | 29.62 | 0.08 |
| Best score | 55 | 1 |
| Average steps | 462.3 | 63.8 |

These are the measured results for the **included checkpoint** (500 training episodes, seed 42; evaluation seed 10,000). A fresh training run may produce different scores. Retrain and evaluate your new checkpoint before presenting new results.

## Project structure

```text
snake-ai-lab/
├── run.bat                     # One-command app launcher
├── train-evaluate-demo.bat      # Train + evaluate + launch GUI
├── setup.ps1                   # First-run environment setup
├── run.py                      # Pygame application entry point
├── train.py                    # DQN training
├── evaluate.py                 # DQN/random evaluation and comparison
├── pyproject.toml
├── requirements.txt
├── src/
│   └── snake_ai_lab/
│       ├── core/                # Snake environment and state encoding
│       ├── ai/                  # DQN, Q-network and replay buffer
│       └── game/                # Pygame interface
├── tests/                       # Automated tests
├── models/checkpoints/
│   └── dqn_snake.pt             # Included pretrained checkpoint
└── logs/training/                # Generated training CSV
```

## Output files

- `models/checkpoints/dqn_snake.pt` — trained weights and checkpoint metadata.
- `logs/training/training.csv` — episode score, steps, reward, loss, epsilon and duration.

The checkpoint is included in this repository so AI Auto Play is available immediately. Training logs are generated locally and excluded from Git.

## Troubleshooting

### Python runtime not found

Install Python 3.12, enable the PATH option in the installer, then close and reopen PowerShell or VS Code. Check:

```powershell
python --version
py -0p
```

### Setup fails during dependency installation

Make sure you have internet access. Delete the incomplete `.venv` folder if setup was interrupted, then run:

```powershell
.\run.bat
```

### AI Auto Play says no checkpoint exists

Train the DQN first using **Train DQN** in the app, or run:

```powershell
.\train-evaluate-demo.bat
```

### Training does not produce high scores

DQN learning is not guaranteed to converge to an optimal policy. Try more training episodes and use evaluation results rather than relying on one game.

## Developer commands

After setup:

```powershell
# Run automated tests
.\.venv\Scripts\python.exe -m pytest

# Train from the terminal
.\.venv\Scripts\python.exe train.py --episodes 500 --seed 42

# Resume training from the saved checkpoint
.\.venv\Scripts\python.exe train.py --episodes 500 --resume

# Evaluate DQN and random baseline
.\.venv\Scripts\python.exe evaluate.py --compare --episodes 100
```

## Current version and future scope

**Current version:** PC-based Python application with Pygame, CPU-based DQN training, checkpointing and evaluation.

ESP32 deployment, browser-based gameplay, and hardware inference are future scope; they are not included in this version.

## Research references

1. V. Mnih et al., “Playing Atari with Deep Reinforcement Learning,” arXiv:1312.5602, 2013.
2. V. Mnih et al., “Human-level control through deep reinforcement learning,” *Nature*, 518, 529–533, 2015.
3. R. S. Sutton and A. G. Barto, *Reinforcement Learning: An Introduction*, 2nd ed., MIT Press, 2018.
