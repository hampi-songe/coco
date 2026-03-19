REGISTRY = {}

from .mpe_episode_runner import EpisodeRunner as MpeEpisodeRunner
REGISTRY["mpeepisode"] = MpeEpisodeRunner

from .episode_runner import EpisodeRunner
REGISTRY["episode"] = EpisodeRunner

from .parallel_runner import ParallelRunner
REGISTRY["parallel"] = ParallelRunner

from .grf_episode_runner import EpisodeRunner as GrfEpisodeRunner
REGISTRY["grfepisode"] = GrfEpisodeRunner