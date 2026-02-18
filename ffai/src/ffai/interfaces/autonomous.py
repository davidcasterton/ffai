"""
Autonomous Mode — controls the ESPN draft room via Playwright browser automation.

WARNING: ESPN's UI can change at any time. All DOM selectors are fragile.
The dead-man's switch will alert and fall back to advisory mode if
selectors fail 10 consecutive polls.

Requires:
    pip install playwright
    playwright install chromium

Usage:
    python scripts/autonomous_draft.py --config config/league.yaml
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from playwright.async_api import async_playwright, Page, BrowserContext
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    logger.warning(
        "Playwright not installed. Autonomous mode unavailable. "
        "Install with: pip install playwright && playwright install chromium"
    )

from ffai.interfaces.advisory import AdvisoryMode
from ffai.interfaces.live_draft_reader import LiveDraftReader
from ffai.rl.state_builder import build_state
from ffai.rl.ppo_agent import PPOAgent
from ffai.value_model.player_value_model import PlayerValueModel

# ESPN draft room selectors — FRAGILE, will break on UI updates
ESPN_SELECTORS = {
    "bid_input": "input[data-testid='bid-amount-input']",
    "bid_button": "button[data-testid='confirm-bid-button']",
    "nominate_search": "input[placeholder='Search players']",
    "player_row": "[data-testid='player-row']",
    "current_bid_display": "[data-testid='current-bid']",
    "nomination_timer": "[data-testid='nomination-timer']",
}

CONSECUTIVE_FAILURE_LIMIT = 10


class AutonomousMode:
    """
    Controls the ESPN draft room autonomously via Playwright.

    Safety features:
    - headless=False so you can monitor / intervene
    - Dead-man's switch: falls back to advisory mode after 10 consecutive
      selector failures
    - All actions logged for audit trail
    """

    def __init__(
        self,
        config_path: Optional[Path] = None,
        value_model: Optional[PlayerValueModel] = None,
        ppo_agent: Optional[PPOAgent] = None,
        rl_team_id: Optional[int] = None,
        budget: float = 200.0,
        poll_interval: float = 2.5,
        headless: bool = False,
    ):
        if not PLAYWRIGHT_AVAILABLE:
            raise ImportError(
                "Playwright is required for autonomous mode. "
                "Install with: pip install playwright && playwright install chromium"
            )

        self.config_path = config_path
        self.value_model = value_model
        self.ppo_agent = ppo_agent
        self.rl_team_id = rl_team_id
        self.budget = budget
        self.headless = headless

        self.advisory = AdvisoryMode(
            config_path=config_path,
            value_model=value_model,
            ppo_agent=ppo_agent,
            rl_team_id=rl_team_id,
            budget=budget,
            poll_interval=poll_interval,
        )

        self._consecutive_failures = 0
        self._fallback_mode = False

    async def _login(self, page: Page) -> bool:
        """Navigate to ESPN and wait for user to log in manually."""
        await page.goto("https://www.espn.com/fantasy/football/")
        logger.info("Please log into ESPN Fantasy Football in the browser window, then press Enter.")
        input("Press Enter after you have logged in...")
        return True

    async def _navigate_to_draft(self, page: Page) -> bool:
        """Navigate to the draft lobby."""
        try:
            await page.goto(
                f"https://fantasy.espn.com/football/league?leagueId={self.advisory.reader.league_id}"
                f"&seasonId={self.advisory.reader.year}"
            )
            logger.info("Navigated to league page. Waiting for draft room...")
            return True
        except Exception as e:
            logger.error(f"Failed to navigate to draft: {e}")
            return False

    async def _get_current_bid(self, page: Page) -> Optional[float]:
        """Read the current bid amount from the draft room UI."""
        try:
            el = await page.query_selector(ESPN_SELECTORS["current_bid_display"])
            if el:
                text = await el.inner_text()
                return float(text.replace("$", "").strip())
        except Exception as e:
            logger.debug(f"Could not read current bid: {e}")
        return None

    async def _place_bid(self, page: Page, amount: float) -> bool:
        """
        Place a bid by filling the input and clicking confirm.

        Returns True on success, False on failure.
        Logs all interactions for audit trail.
        """
        try:
            bid_input = await page.query_selector(ESPN_SELECTORS["bid_input"])
            if not bid_input:
                logger.warning("Bid input not found")
                return False

            await bid_input.fill(str(int(amount)))
            logger.info(f"ACTION: Filled bid input with ${int(amount)}")

            bid_button = await page.query_selector(ESPN_SELECTORS["bid_button"])
            if not bid_button:
                logger.warning("Bid confirm button not found")
                return False

            await bid_button.click()
            logger.info(f"ACTION: Clicked bid confirm for ${int(amount)}")
            self._consecutive_failures = 0
            return True

        except Exception as e:
            self._consecutive_failures += 1
            logger.error(f"Failed to place bid: {e} (failure #{self._consecutive_failures})")

            if self._consecutive_failures >= CONSECUTIVE_FAILURE_LIMIT:
                logger.critical(
                    f"DEAD-MAN'S SWITCH: {CONSECUTIVE_FAILURE_LIMIT} consecutive "
                    "selector failures. Falling back to advisory mode."
                )
                self._fallback_mode = True

            return False

    async def _nominate_player(self, page: Page, player_name: str) -> bool:
        """Search for and nominate a player."""
        try:
            search = await page.query_selector(ESPN_SELECTORS["nominate_search"])
            if not search:
                logger.warning("Nomination search input not found")
                return False

            await search.fill(player_name)
            await asyncio.sleep(0.5)

            # Find and click the first matching player row
            rows = await page.query_selector_all(ESPN_SELECTORS["player_row"])
            if rows:
                await rows[0].click()
                logger.info(f"ACTION: Nominated {player_name}")
                self._consecutive_failures = 0
                return True

        except Exception as e:
            self._consecutive_failures += 1
            logger.error(f"Failed to nominate player: {e} (failure #{self._consecutive_failures})")

            if self._consecutive_failures >= CONSECUTIVE_FAILURE_LIMIT:
                logger.critical("DEAD-MAN'S SWITCH: falling back to advisory mode.")
                self._fallback_mode = True

        return False

    async def run(self):
        """Main autonomous draft loop."""
        if self.rl_team_id is None:
            raise ValueError("rl_team_id must be set before running autonomous mode.")

        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=self.headless)
            context = await browser.new_context()
            page = await context.new_page()

            await self._login(page)
            await self._navigate_to_draft(page)

            logger.info("Autonomous mode active. Monitoring draft...")

            for sim_state in self.advisory.reader.poll_loop(
                rl_team_id=self.rl_team_id,
                budget=self.budget,
            ):
                if self._fallback_mode:
                    # Fall back to just printing recommendations
                    available = sim_state.get("_available_players", [])
                    if available:
                        top = self.advisory._score_players(available, sim_state)
                        self.advisory._print_recommendation(top, sim_state, pick_num=0)
                    continue

                # Get bid recommendation from PPO agent
                available = sim_state.get("_available_players", [])
                if not available:
                    continue

                top_players = self.advisory._score_players(available, sim_state)
                if not top_players:
                    continue

                best_player = top_players[0]
                state_tensor = build_state(sim_state, current_player=best_player, current_bid=0.0)

                current_budget = float(sim_state.get("rl_team_budget", self.budget))
                bid_amount, _, _ = self.ppo_agent.get_bid_action(
                    state_tensor, current_budget, min_bid=1.0, deterministic=True
                )

                current_bid = await self._get_current_bid(page) or 0.0
                if bid_amount > current_bid:
                    await self._place_bid(page, bid_amount)

            await browser.close()
