from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class OptimizerSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="OPTIMIZER_")

    num_threads: int | None = Field(default=None, description="Number of threads to use for optimization")
    solver: Literal['cbc', 'highs', 'hybrid'] = Field(default='cbc',
                                                      description="MILP solver: cbc, the CBC binary pulp resolves; highs, HiGHS "
                                                      "in process through highspy; or hybrid, CBC for the joint probe and "
                                                      "HiGHS for the cost, tie break and continuity stages")
    cbc_path: str | None = Field(default=None,
                                 description="CBC executable. Unset uses the one pulp resolves, a cbc on PATH")
    gap_abs: float | None = Field(default=0.02,
                                  description="Absolute MIP gap in currency units, applied to the cost stage. The solver "
                                              "stops once the remaining gap is worth less than this, two cents by default. "
                                              "Unset solves to proven optimality")
    gap_share: float = Field(default=0.0025,
                             description="Share of the money a request moves that the absolute gap may grow to: the "
                                         "goal energy of its batteries at the mean import price. A fixed cent sits below "
                                         "the slack the c_min gate leaves in the relaxation on vehicles with a charge goal, "
                                         "see #186. Zero keeps gap_abs fixed")
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
    blocked_clients: str = Field(default="evcc/0.304.1",
                                 description="Comma separated User-Agents answered with 400 and an upgrade notice instead of a solve. "
                                             "evcc 0.304.1 sends 400 to 480 step horizons the solver cannot close: 65 percent of its "
                                             "requests came back Not Solved after burning the probe and cost stage caps")
    dump_slow_requests: str | None = Field(default=None,
                                           description="JSON Lines file requests that exhaust the solver time limit, leave more than one currency "
                                                       "unit unproven in the cost stage, or come back without a schedule are appended to, with the token "
                                                       "subject. Unset disables the dump")
    azure_openai_endpoint: str | None = Field(default=None,
                                              description="Azure OpenAI resource the plan explanations are generated with, such as "
                                                          "https://name.openai.azure.com/. Unset disables /optimize/explain")
    azure_openai_deployment: str = Field(default="gpt-6.1-sol", description="Deployment name of the chat model behind /optimize/explain")
    azure_openai_key: str | None = Field(default=None, description="API key of the Azure OpenAI resource")
    feedback_table_endpoint: str | None = Field(default=None,
                                                description="Azure table service the explanations and the votes on them are stored in, "
                                                            "such as https://name.table.core.windows.net, accessed with the managed identity. "
                                                            "Unset without a connection string disables /optimize/explain/feedback")
    feedback_connection_string: str | None = Field(default=None,
                                                   description="Storage account connection string for the explanations table, for "
                                                               "local runs and Azurite. Takes precedence over the endpoint")
