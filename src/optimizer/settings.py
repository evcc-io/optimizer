from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class OptimizerSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="OPTIMIZER_")

    num_threads: int | None = Field(default=None, description="Number of threads to use for optimization")
    gap_abs: float | None = Field(default=0.01,
                                  description="Absolute MIP gap in currency units, applied to the cost stage. The solver "
                                              "stops once the remaining gap is worth less than this, one cent by default. "
                                              "Unset solves to proven optimality")
    probe_seconds: float | None = Field(default=None,
                                        description="Seconds the joint solve gets before falling back to the two "
                                                    "stage split. Unset derives it from the time limit. Zero always "
                                                    "splits, which is the behaviour before this became adaptive")
    preference_budget: float = Field(default=0.0,
                                     description="Real money the strategy stage may spend on preferences, in currency "
                                                 "units. Zero keeps the strategies cost neutral as documented, they then "
                                                 "only choose between equally priced schedules, see #114")
    time_limit: float | None = Field(default=None, description="Time limit for the optimization process in seconds")
    log_subject: bool = Field(default=False, description="Log the JWT subject of every request. Off by default, it names accounts in the logs")
    dump_slow_requests: str | None = Field(default=None,
                                           description="JSON Lines file requests that exhaust the solver time limit are appended to. Unset disables the dump")
