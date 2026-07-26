from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class OptimizerSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="OPTIMIZER_")

    num_threads: int | None = Field(default=None, description="Number of threads to use for optimization")
    gap_abs: float | None = Field(default=0.01,
                                  description="Absolute MIP gap in currency units. The solver stops once the remaining "
                                              "gap is worth less than this, one cent by default. Unset solves to proven "
                                              "optimality")
    strategy_weight: float = Field(default=3.0,
                                   description="Multiplier on the cost neutral strategy terms. Raises them out of the "
                                               "solver's tolerance band so ties get decided instead of searched, see #114")
    time_limit: float | None = Field(default=None, description="Time limit for the optimization process in seconds")
    log_subject: bool = Field(default=False, description="Log the JWT subject of every request. Off by default, it names accounts in the logs")
    dump_slow_requests: str | None = Field(default=None,
                                           description="JSON Lines file requests that exhaust the solver time limit are appended to. Unset disables the dump")
