"""
AuctionDraftEnv — PufferLib native PufferEnv for auction draft RL.

Wraps AuctionDraftSimulator via the draft_steps() generator interface.
One RL step = one bidding decision by the RL team.

Episode lifecycle (auto-reset, handled internally per PufferEnv contract):
  reset()  → primes generator, stores first obs
  step(a)  → sends bid fraction, collects reward, advances to next RL turn
           → on StopIteration (draft done), optionally runs season sim,
             then immediately resets to start of next episode

Data loading:
  AuctionDraftSimulator loads ESPN CSV data from disk on every construction.
  _SimDataCache caches it at the class level (once per process) so that
  repeated episode resets within one worker are fast.
"""

import logging
from pathlib import Path

import numpy as np
import gymnasium

import pufferlib

from ffai.simulation.auction_draft_simulator import AuctionDraftSimulator, _build_team_manager_map
from ffai.rl.state_builder import build_state, STATE_DIM
from ffai.rl.reward import terminal_reward, normalize_terminal_reward
from ffai.data.espn_scraper import load_league_config as _load_league_config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-process data cache
# ---------------------------------------------------------------------------

class _SimDataCache:
    """Class-level cache so each worker process loads ESPN data only once."""
    _store: dict = {}  # keyed by year

    @classmethod
    def get(cls, year: int):
        if year not in cls._store:
            logger.info(f"_SimDataCache: loading data for year {year}")
            from ffai.data.espn_scraper import ESPNDraftScraper
            scraper = ESPNDraftScraper()
            cls._store[year] = scraper.load_or_fetch_data(year)
        return cls._store[year]


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class AuctionDraftEnv(pufferlib.PufferEnv):
    """
    Native PufferEnv wrapping AuctionDraftSimulator via draft_steps() generator.

    Observation space: Box(56,)  — flat state vector from state_builder
    Action space:      Box(1,)   — bid fraction ∈ [0, 1], decoded as
                                   max_bid = fraction * remaining_budget
    """

    def __init__(
        self,
        year: int = 2024,
        budget: int = 200,
        enable_season_sim: bool = False,
        season_sim_interval: int = 10,
        buf=None,
        seed=None,
    ):
        self.year = year
        self.budget = budget
        self.enable_season_sim = enable_season_sim
        self.season_sim_interval = season_sim_interval
        self._episode_count = 0
        self._sim = None
        self._gen = None

        # PufferEnv requires these set BEFORE super().__init__()
        self.num_agents = 1
        self.single_observation_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=(STATE_DIM,), dtype=np.float32
        )
        self.single_action_space = gymnasium.spaces.Box(
            low=0.0, high=1.0, shape=(1,), dtype=np.float32
        )

        super().__init__(buf=buf)

    # ------------------------------------------------------------------
    # Simulator construction
    # ------------------------------------------------------------------

    def _make_simulator(self) -> AuctionDraftSimulator:
        """
        Create a new AuctionDraftSimulator, reusing cached ESPN data so that
        disk reads happen only once per worker process.
        """
        draft_df, stats_df, weekly_df, predraft_df, settings = _SimDataCache.get(self.year)

        # Create simulator without triggering its own data load by bypassing
        # __init__ and manually setting attributes.
        sim = AuctionDraftSimulator.__new__(AuctionDraftSimulator)
        sim.year = self.year
        sim.budget = self.budget
        sim.rl_team_name = "Team 1"
        sim.rl_model = None
        sim.draft_completed = False
        _cfg = _load_league_config()
        _league_name = _cfg["league"]["league_name"]
        _league_id = _cfg["league"]["league_id"]
        sim.data_dir = Path(__file__).parent.parent / f"data/{_league_name}"

        # Inject cached data
        sim.draft_df = draft_df
        sim.stats_df = stats_df
        sim.weekly_df = weekly_df
        sim.predraft_df = predraft_df
        sim.settings = settings

        # Run the cheap init steps
        sim.teams = sim.initialize_teams()
        sim.nomination_order = list(sim.teams.keys())
        sim.available_players = sim.initialize_available_players()
        sim.players_drafted = []

        # Feature store and manager tendencies (set once, reused across episodes)
        if not hasattr(self, '_feature_store'):
            try:
                from ffai.data.feature_store import FeatureStore
                import csv
                fs = FeatureStore()
                mgr_tend = {}
                mt_path = Path(__file__).parent.parent / f"data/{_league_name}_processed/manager_tendencies_{_league_id}.csv"
                if mt_path.exists():
                    with open(mt_path) as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            mgr_tend[row["manager_id"]] = row
                self._feature_store = fs
                self._mgr_tend = mgr_tend
            except Exception:
                self._feature_store = None
                self._mgr_tend = {}

        sim.feature_store = self._feature_store
        sim._manager_tendencies = self._mgr_tend

        # Build "Team N" → manager_id mapping (same logic as AuctionDraftSimulator.__init__)
        sim._team_manager_map = _build_team_manager_map(draft_df)

        return sim

    # ------------------------------------------------------------------
    # Episode management
    # ------------------------------------------------------------------

    def _start_episode(self) -> np.ndarray:
        """Create new simulator + generator, prime to first RL decision."""
        self._sim = self._make_simulator()
        self._sim._step_reward = 0.0
        self._gen = self._sim.draft_steps()

        try:
            state, player, bid = next(self._gen)
            obs = self._build_obs(state, player, bid)
        except StopIteration:
            # Extremely unlikely: draft had no RL decisions at all.
            logger.warning("AuctionDraftEnv: draft had no RL decisions; returning zero obs.")
            obs = np.zeros(STATE_DIM, dtype=np.float32)

        return obs

    def _build_obs(self, state: dict, player: dict, current_bid: float) -> np.ndarray:
        feature_store = getattr(self._sim, 'feature_store', None)
        return build_state(
            state,
            current_player=player,
            current_bid=current_bid,
            feature_store=feature_store,
            year=self.year,
        ).numpy()

    # ------------------------------------------------------------------
    # PufferEnv interface
    # ------------------------------------------------------------------

    def reset(self, seed=None):
        obs = self._start_episode()
        self.observations[:] = obs
        self.rewards[:] = 0.0
        self.terminals[:] = False
        self.truncations[:] = False
        return self.observations, [{}]

    def step(self, actions):
        # Decode action: bid fraction → max willing to pay
        action_frac = float(np.clip(actions.flatten()[0], 0.0, 1.0))
        budget = float(self._sim.teams[self._sim.rl_team_name]["current_budget"])
        max_bid = action_frac * budget

        try:
            state, player, bid = self._gen.send(max_bid)
            step_reward = float(self._sim._step_reward)
            terminal = False
            info = {}
            obs = self._build_obs(state, player, bid)

        except StopIteration:
            step_reward = float(self._sim._step_reward)
            terminal = True

            # Optional terminal reward from season simulation
            if self.enable_season_sim:
                self._episode_count += 1
                if self._episode_count % self.season_sim_interval == 0:
                    step_reward += self._run_season_sim()

            info = {
                "draft_completed": int(self._sim.draft_completed),
                "draft_reward": step_reward,
            }

            # Auto-reset: start next episode, store its first obs
            obs = self._start_episode()

        self.observations[:] = obs
        self.rewards[:] = step_reward
        self.terminals[:] = terminal
        self.truncations[:] = False

        return self.observations, self.rewards, self.terminals, self.truncations, [info]

    def close(self):
        pass

    # ------------------------------------------------------------------
    # Season simulation
    # ------------------------------------------------------------------

    def _run_season_sim(self) -> float:
        """Run season sim on completed draft, return normalized terminal reward."""
        try:
            from ffai.simulation.season_simulator import SeasonSimulator

            draft_results = self._sim.get_draft_results()
            season_sim = SeasonSimulator(
                draft_results=draft_results["teams"],
                year=self.year,
            )
            season_sim.simulate_season()
            standings = season_sim.get_standings()  # sorted list of (team, wins)

            standing_pos = next(
                i for i, (team, _) in enumerate(standings)
                if team == self._sim.rl_team_name
            )
            wins = season_sim.standings[self._sim.rl_team_name]
            raw = terminal_reward(standing_pos, wins)
            return normalize_terminal_reward(raw)

        except Exception as exc:
            logger.warning(f"Season sim failed: {exc}")
            return 0.0
