from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class OptimizerSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="OPTIMIZER_")

    num_threads: int | None = Field(default=None, description="Number of threads to use for optimization")
    time_limit: float | None = Field(default=None, description="Time limit for the optimization process in seconds")
    log_subject: bool = Field(default=False, description="Log the JWT subject of every request. Off by default, it names accounts in the logs")
    dump_slow_requests: bool = Field(default=False, description="Write requests that exhaust the solver time limit to disk so they can be replayed")
    dump_dir: str = Field(default="/tmp/slow-requests", description="Directory the slow request dumps are written to")
