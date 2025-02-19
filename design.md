# Fantasy Football Auction Draft and Season Simulator, and Auction Draft RL Model Trainer

## 1. **Project Overview**

This project aims to develop a **fantasy football auction draft and season simulator,** and a **reinforcement learning (RL) model** trained using PyTorch to optimize draft strategies. The system will:

- Simulate a **12-team auction draft** following ESPN fantasy league rules.
- Simulate **weekly matchups** using historical fantasy scoring.
- Train an RL model to **learn optimal draft strategies** that maximize weekly wins and end of season standings.

---

## 2. **System Components**

### 2.1 **Auction Draft Simulator**

- Implements **12-team auction draft logic**.
- Uses **historical draft results** to model realistic team manager behavior.
- Outputs **team rosters** after the draft.

### 2.2 **Season Simulator**

- Uses **historical weekly performance data** to simulate **matchups**.
- Calculates **weekly fantasy points** for each team.
- Determines **weekly winners** based on scoring rules.

### 2.3 **Reinforcement Learning (RL) Model**

- Uses **auction simulator and season simulator** as an environment.
- Implements **state, actions, and rewards** to train a drafting agent.
- Optimizes draft strategies to **maximize weekly wins**.

---

## 3. **Design Details**

### 3.1 **Auction Draft Simulator**

#### **Inputs:**

- **Team Managers**: Names of team managers participating in the draft
- **Scoring Rules**: How players score points each week.
- **Roster Rules**: Available roster slots, bench slots, and rules for max players that can be drafter from a given position.
- **Player Data**:&#x20;
  - List of all players that can be drafted
  - Prediction of each players stats for the upcoming year
  - Prediction of each players total points for the upcoming year
  - Prediction of each players value in the draft
- **Budget constraints**: Each team has \$200.

**Setup:**

- 11 team managers will bid using ESPN's auto-draft logic.&#x20;
  - I believe this logic bids up to ESPN's default predicted value of each player, provided the team's budget allows. Please make this logic as close as possible to how auto-draft works in ESPN auction drafts.
- The 12th team will run the RL model being trained to optimize its auction strategy to maximize the number of weeks it wins in the season simulator.
  - Through iterations of running the auction and season simulator, this RL model should learn how to improve it's auction drafting  to maximize the number of weeks it wins during the season simulator.

**Logic:**

- For our auction draft simulator, we will have 11 teams using the ESPN auto-draft logic, and 1 team training an RL model to optimize it's auction strategy.
- The RL model's goal is to learn how to draft the team within it's budget that will win the most weeks in the season simulator.
- The high level logic of the auction draft is as follows:
  - The auction rotates which team is nominating a player in order.
  - When a team nominates a player, they place a bid of $1 for that player.
  - Each team decides the max value they will bid for the nominated player.
    - Each auto-draft team decides the max value they are willing to pay for a player by using the predicted value of the player +/- a random 10% of that value, rounded to the nearest $1.
    - The RL team will use its model to decide the max value for the nominated player.
  - Each team gets to decide if they want to bid $1 more than the previous team.
  - Bidding for a player stops when 11 teams decide to stop bidding for that player.
  - The team with the highest bid for that player adds them to their roster.
  - The auction then rotates to the next team and the process repeats.
  - The auction ends when all 12 teams have completed their rosters.
  - If the RL model cannot complete a full roster within it's budget, then the season simulator should stop and the model should be updated to try to improve it's strategy.
  - All teams are required to complete exactly a full roster. They cannot have more or less players than the defined roster slots.
  - Teams cannot exceed their budget of $200.

#### **Outputs:**

- **Team Rosters:** the rosters of all teams coming out of the auction draft, the order players were drafted, and how much was spent on each player

---

### 3.2 **Season Simulator**

#### **Inputs:**

- **Draft results**: each teams roster
- **Matchup schedule:** which managers are playing each other each week, this is randomly  defined before the season.
- **Predicted player points per week:** predicted points that each player will score  each week
- **Actual player points per week:** actual points that each player scored each week

#### **Logic:**

- **Weekly matchups:** Each team faces another team each week.
- **Roster Selection:** Each team picks where to place it's players across it's roster slots and bench based on the predicted points each player will score that week. Teams move players that have higher point predictions into roster slots, and lower point predictions into bench slots.
- **Scoring:** The points each team scores for the week is the sum of actual points scored by players in non-bench roster slots. Players in bench slots to not count toward weekly scoring.
- **Win/loss determination:** The team with the higher sum of points wins the week.

#### **Outputs:**

- **Season standings**: a ranked list of teams, the team that wins the most weeks is #1 and the remaining managers are ranked in descending order of the number of weeks they won. If multiple teams won the same number of weeks, then they have tied for a given order in the standings.
- **Matchup results per week:** a win/loss calculation for each team each week, and the number of points each team scored that week

---

### 3.3 **Reinforcement Learning Model**

#### **Problem Formulation:**

- **State (S):**
  - RL teams remaining budget
  - Opponent teams remaining budgets
  - Draft turn, based on number of players that have been drafted
  - Predicted points per roster slot on each team
  - Auction \$ spent per roster slot on each team
  - Nominated player
  - Nominated players  predicted pre-draft value
  - Nominated players predicted points for the season
- **Actions (A):**
  - Max bid amount for nominated player
  - Pass on bidding
- **Reward (R):**
  - Weekly win = +1.
  - 1st place in season standings = +10
  - 2nd place in season standings = +5
  - 3rd or 4th in season standings = +2
  - Cannot complete a roster within budget = -10

#### **Training Process:**

1. Auction drafts + season simulations are played thousands of times.
2. RL agent adjusts auction bidding strategy based on simulated season outcomes.
3. Model optimizes draft choices to maximize weekly wins and position in season standings

---

## 4. **Technology Stack**

- **Backend**: Python, Flask (for API interactions)
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: PyTorch, Reinforcement Learning (DQN or PPO)
- **Simulation**: Custom fantasy football draft & season engine

---

## 5. **Next Steps**

1. Implement **auction draft simulator**.
2. Implement **season simulator**.
3. Train RL model using **simulated drafts & seasons**.
4. Evaluate **draft strategies learned by RL agent**.

---

This project will provide **an AI-driven strategy guide** for fantasy football drafts, optimizing for **weekly wins over an entire season**.

