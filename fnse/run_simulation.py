"""
Integration script to run a complete FNSE simulation.

This script:
1. Initializes a simulation epoch with specified configuration
2. Runs the simulation tick-by-tick (or to completion)
3. Monitors progress and logs key metrics
4. Outputs final results and statistics
"""

import asyncio
import json
import logging
import sys
import time
from datetime import UTC, datetime
from typing import Any

from fnse.config import settings
from fnse.engine.macro_swarm import AgentRole, MacroSwarm, SwarmConfig, swarm_manager
from fnse.engine.safeguards import AlertSeverity, SafeguardSystem

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class SimulationRunner:
    """Runs and monitors a fractal neural swarm simulation."""

    def __init__(
        self,
        num_agents: int = 10,
        max_ticks: int = 100,
        global_objective: str = "minimize_loss",
        loss_function: str = "mse",
        convergence_threshold: float = 0.01,
        checkpoint_interval: int = 10,
        agent_roles: list | None = None,
        verbose: bool = True,
    ):
        self.num_agents = num_agents
        self.max_ticks = max_ticks
        self.global_objective = global_objective
        self.loss_function = loss_function
        self.convergence_threshold = convergence_threshold
        self.checkpoint_interval = checkpoint_interval
        self.agent_roles = agent_roles
        self.verbose = verbose

        self.epoch_id: str | None = None
        self.swarm: MacroSwarm | None = None
        self.safeguard: SafeguardSystem | None = None
        self.start_time: float | None = None
        self.tick_results: list = []

    def initialize(self) -> str:
        """Initialize a new simulation epoch."""
        # Parse agent roles
        roles = []
        if self.agent_roles:
            for role_str in self.agent_roles:
                try:
                    roles.append(AgentRole(role_str))
                except ValueError:
                    logger.warning(f"Invalid agent role: {role_str}, using defaults")

        config = SwarmConfig(
            epoch_id=f"epoch_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}",
            num_agents=self.num_agents,
            agent_roles=roles
            or [
                AgentRole.EXPLORER,
                AgentRole.OPTIMIZER,
                AgentRole.CRITIC,
                AgentRole.SYNTHESIZER,
                AgentRole.COORDINATOR,
            ],
            max_ticks=self.max_ticks,
            global_objective=self.global_objective,
            loss_function=self.loss_function,
            convergence_threshold=self.convergence_threshold,
            checkpoint_interval=self.checkpoint_interval,
        )

        # Create swarm (initializes agents)
        self.swarm = swarm_manager.create_swarm(config)
        self.epoch_id = config.epoch_id

        # Initialize safeguard system
        self.safeguard = SafeguardSystem(
            epoch_id=self.epoch_id, checkpoint_interval=self.checkpoint_interval
        )

        self._log(f"Initialized epoch: {self.epoch_id}")
        self._log(f"Agents: {len(self.swarm.agents)}, Max ticks: {self.max_ticks}")
        self._log(f"Objective: {self.global_objective}, Loss: {self.loss_function}")

        return self.epoch_id

    def _log(self, message: str, level: str = "info"):
        """Log a message if verbose mode is enabled."""
        if self.verbose:
            getattr(logger, level)(message)

    def run_tick(self) -> dict[str, Any]:
        """Execute a single simulation tick and return metrics."""
        if not self.swarm:
            raise RuntimeError("Simulation not initialized. Call initialize() first.")

        # Execute tick
        packet = self.swarm.tick()

        # Check safeguards
        alerts = self.safeguard.on_tick_end(packet) if self.safeguard else []

        # Log critical alerts
        for alert in alerts:
            if alert.severity in (AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY):
                self._log(f"ALERT [{alert.severity.value}]: {alert.message}", "warning")

        # Record results
        tick_result = {
            "tick": packet.tick_number,
            "global_loss": packet.global_loss,
            "convergence_rate": packet.convergence_rate,
            "agent_count": len(packet.agent_states),
            "alerts_count": len(alerts),
            "timestamp": packet.timestamp.isoformat(),
        }
        self.tick_results.append(tick_result)

        self._log(
            f"Tick {packet.tick_number:4d} | "
            f"Loss: {packet.global_loss:.6f} | "
            f"Conv: {packet.convergence_rate:.6f} | "
            f"Agents: {len(packet.agent_states)} | "
            f"Alerts: {len(alerts)}"
        )

        return tick_result

    def cleanup(self):
        """Clean up resources."""
        if self.swarm:
            try:
                self.swarm._running = False
            except (RuntimeError, ValueError, KeyError, TypeError, AttributeError) as e:
                self._log(f"Error stopping swarm: {e}", "warning")
        if self.safeguard:
            try:
                self.safeguard.emergency_stop()
            except (RuntimeError, ValueError, KeyError, TypeError, AttributeError) as e:
                self._log(f"Error shutting down safeguard: {e}", "warning")

    def save_results(self, result: dict[str, Any], filepath: str):
        """Save simulation results to a JSON file."""
        output = {
            "epoch_id": result["epoch_id"],
            "converged": result["converged"],
            "final_global_loss": result["final_global_loss"],
            "total_ticks": result["total_ticks"],
            "duration_seconds": result["total_duration_seconds"],
            "agent_count": len(result.get("agent_statistics", {})),
            "agent_states": result.get("agent_statistics", {}),
            "tick_history": self.tick_results,
            "config": {
                "num_agents": self.num_agents,
                "max_ticks": self.max_ticks,
                "global_objective": self.global_objective,
                "loss_function": self.loss_function,
                "convergence_threshold": self.convergence_threshold,
                "checkpoint_interval": self.checkpoint_interval,
                "agent_roles": self.agent_roles,
            },
        }
        with open(filepath, "w") as f:
            json.dump(output, f, indent=2, default=str)
        self._log(f"Results saved to {filepath}")

    def run_to_completion(self) -> dict[str, Any]:
        """Run simulation until convergence or max ticks reached."""
        if not self.swarm:
            raise RuntimeError("Simulation not initialized. Call initialize() first.")

        self.start_time = time.time()
        self._log(f"Starting simulation for epoch {self.epoch_id}")

        converged = False

        for tick_num in range(1, self.max_ticks + 1):
            result = self.run_tick()

            # Check convergence
            if result["global_loss"] < self.convergence_threshold:
                self._log(
                    f"Converged at tick {tick_num} with loss {result['global_loss']:.6f}"
                )
                converged = True
                break

            # Check if swarm stopped running
            if not self.swarm._running:
                self._log("Swarm stopped running")
                break

        duration = time.time() - self.start_time
        final_result = self.get_final_result(converged, duration)

        self._log(f"Simulation completed in {duration:.2f}s")
        self._log(f"Final loss: {final_result['final_global_loss']:.6f}")
        self._log(f"Converged: {final_result['converged']}")

        return final_result

    def get_final_result(self, converged: bool, duration: float) -> dict[str, Any]:
        """Get comprehensive final results."""
        if not self.swarm:
            raise RuntimeError("Simulation not initialized.")

        result = self.swarm.get_epoch_result()

        # Agent statistics
        agent_stats: dict[str, dict[str, int | float | str]] = {}
        for agent_id, agent_node in self.swarm.agents.items():
            state = agent_node.state
            agent_stats[agent_id] = {
                "role": state.role.value,
                "status": state.status.value,
                "ticks": state.tick_count,
                "successes": state.success_count,
                "failures": state.failure_count,
                "divergence": state.divergence_score,
                "tokens": state.total_tokens_used,
            }

        # Top performers
        top_performers = sorted(
            agent_stats.items(),
            key=lambda x: int(x[1]["successes"]) - int(x[1]["failures"]),
            reverse=True,
        )[:5]
        return {
            "epoch_id": self.epoch_id,
            "converged": converged,
            "final_global_loss": result.final_global_loss,
            "total_ticks": result.total_ticks,
            "loss_trajectory": result.loss_trajectory,
            "total_duration_seconds": duration,
            "total_tokens": result.total_tokens,
            "top_performers": [p[0] for p in top_performers],
            "skills_compiled": len(result.skills_compiled),
            "circuit_breaks": result.circuit_breaks,
            "rollbacks_performed": result.rollbacks_performed,
            "agent_statistics": agent_stats,
            "checkpoints_created": len(self.swarm.checkpoints),
        }

    def print_summary(self, result: dict[str, Any]):
        """Print a formatted summary of the simulation results."""
        print("\n" + "=" * 60)
        print(f"SIMULATION SUMMARY - {result['epoch_id']}")
        print("=" * 60)
        print(f"Converged:           {result['converged']}")
        print(f"Final Global Loss:   {result['final_global_loss']:.6f}")
        print(f"Total Ticks:         {result['total_ticks']}")
        print(f"Duration:            {result['total_duration_seconds']:.2f}s")
        print(f"Total Tokens Used:   {result['total_tokens']}")
        print(f"Skills Compiled:     {result['skills_compiled']}")
        print(f"Circuit Breaks:      {result['circuit_breaks']}")
        print(f"Rollbacks:           {result['rollbacks_performed']}")
        print(f"Checkpoints:         {result['checkpoints_created']}")

        print("\nTop 5 Performers:")
        for i, agent_id in enumerate(result["top_performers"], 1):
            stats = result["agent_statistics"][agent_id]
            print(
                f"  {i}. {agent_id} ({stats['role']}) - "
                f"S:{stats['successes']} F:{stats['failures']} "
                f"D:{stats['divergence']:.3f} T:{stats['tokens']}"
            )


async def run_async_simulation(
    num_agents: int = 10,
    max_ticks: int = 100,
    global_objective: str = "minimize_loss",
    loss_function: str = "mse",
    convergence_threshold: float = 0.01,
    checkpoint_interval: int = 10,
    agent_roles: list | None = None,
    verbose: bool = True,
    output_file: str | None = None,
) -> dict[str, Any]:
    """Run simulation asynchronously."""
    runner = SimulationRunner(
        num_agents=num_agents,
        max_ticks=max_ticks,
        global_objective=global_objective,
        loss_function=loss_function,
        convergence_threshold=convergence_threshold,
        checkpoint_interval=checkpoint_interval,
        agent_roles=agent_roles,
        verbose=verbose,
    )

    try:
        runner.initialize()
        result = runner.run_to_completion()

        if verbose:
            runner.print_summary(result)

        if output_file:
            runner.save_results(result, output_file)

        return result
    finally:
        runner.cleanup()


def run_sync_simulation(
    num_agents: int = 10,
    max_ticks: int = 100,
    global_objective: str = "minimize_loss",
    loss_function: str = "mse",
    convergence_threshold: float = 0.01,
    checkpoint_interval: int = 10,
    agent_roles: list | None = None,
    verbose: bool = True,
    output_file: str | None = None,
) -> dict[str, Any]:
    """Run simulation synchronously (blocking)."""
    return asyncio.run(
        run_async_simulation(
            num_agents=num_agents,
            max_ticks=max_ticks,
            global_objective=global_objective,
            loss_function=loss_function,
            convergence_threshold=convergence_threshold,
            checkpoint_interval=checkpoint_interval,
            agent_roles=agent_roles,
            verbose=verbose,
            output_file=output_file,
        )
    )


def main():
    """Main entry point with CLI argument parsing."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run FNSE Fractal Neural Swarm Simulation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--agents", "-a", type=int, default=10, help="Number of agents in the swarm"
    )
    parser.add_argument(
        "--ticks",
        "-t",
        type=int,
        default=100,
        help="Maximum number of simulation ticks",
    )
    parser.add_argument(
        "--objective",
        "-o",
        type=str,
        default="minimize_loss",
        help="Global objective for the swarm",
    )
    parser.add_argument(
        "--loss",
        "-l",
        type=str,
        default="mse",
        choices=["mse", "mae", "cosine", "custom"],
        help="Loss function to use",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.01, help="Convergence threshold"
    )
    parser.add_argument(
        "--checkpoint", "-c", type=int, default=10, help="Checkpoint interval"
    )
    parser.add_argument(
        "--roles",
        "-r",
        nargs="+",
        choices=[r.value for r in AgentRole],
        help="Specific agent roles to use",
    )
    parser.add_argument(
        "--output", "-O", type=str, help="Output file for results (JSON)"
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true", help="Suppress verbose output"
    )

    args = parser.parse_args()

    # Run simulation
    result = run_sync_simulation(
        num_agents=args.agents,
        max_ticks=args.ticks,
        global_objective=args.objective,
        loss_function=args.loss,
        convergence_threshold=args.threshold,
        checkpoint_interval=args.checkpoint,
        agent_roles=args.roles,
        verbose=not args.quiet,
        output_file=args.output,
    )

    # Exit with appropriate code
    sys.exit(0 if result["converged"] else 1)


if __name__ == "__main__":
    main()
