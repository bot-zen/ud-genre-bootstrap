"""Bootstrap scheduling algorithm."""

import logging
from typing import Dict, List, Set, Tuple

logger = logging.getLogger(__name__)


class BootstrapScheduler:
    """Create bootstrapping schedule for progressive genre labeling."""

    def __init__(self, max_iterations: int = 10):
        """Initialize scheduler.

        Args:
            max_iterations: Maximum number of bootstrap iterations
        """
        self.max_iterations = max_iterations

    def create_schedule(
        self, genre_combinations: Set[Tuple[str, ...]]
    ) -> List[Dict[str, List]]:
        """Create bootstrap schedule from genre combinations.

        Based on the algorithm from Müller-Eberstein et al. (2021).

        Args:
            genre_combinations: Set of genre tuples from treebanks
                                e.g., {('news',), ('blog', 'fiction'), ...}

        Returns:
            List of environments, each containing:
                - 'known': List of genres we can use as anchors
                - 'predict': List of genre combinations we can partially label
                - 'disjunct': List of combinations we cannot yet resolve

        Raises:
            ValueError: If schedule cannot resolve all combinations
        """
        schedule = []

        # Extract all unique genres
        all_genres = {g for combo in genre_combinations for g in combo}

        # Start with single-genre treebanks as known
        known_genres = {combo[0] for combo in genre_combinations if len(combo) == 1}
        known_combinations = {combo for combo in genre_combinations if len(combo) == 1}

        prev_num_known = -1
        iteration = 0

        logger.info(f"Starting with {len(known_genres)} known genres: {sorted(known_genres)}")
        logger.info(f"Total genre combinations to resolve: {len(genre_combinations)}")
        logger.info(f"All unique genres in dataset: {sorted(all_genres)}")

        # Log distribution of genre combination sizes
        combo_sizes = {}
        for combo in genre_combinations:
            size = len(combo)
            combo_sizes[size] = combo_sizes.get(size, 0) + 1
        logger.info(f"Genre combination distribution: {dict(sorted(combo_sizes.items()))}")

        # Iterate until no more progress or max iterations
        while prev_num_known != len(known_genres) and iteration < self.max_iterations:
            environment = {"known": [], "predict": [], "disjunct": []}
            prev_num_known = len(known_genres)

            # Known genres act as anchors
            environment["known"] = sorted(known_genres)

            # Predict: combinations with at least one known genre
            predict_combinations = {
                combo
                for combo in genre_combinations
                if (len(set(combo) & known_genres) > 0) and (combo not in known_combinations)
            }
            environment["predict"] = sorted(predict_combinations)

            # Disjunct: combinations with no known genres
            unknown_combinations = {
                combo
                for combo in genre_combinations
                if len(set(combo) & known_genres) == 0
            }
            environment["disjunct"] = sorted(unknown_combinations)

            # Determine newly resolvable genres
            # A genre becomes known when it's the only unknown in a combination
            new_known_genres = {
                (set(combo) - known_genres).pop()
                for combo in genre_combinations
                if len(set(combo) - known_genres) == 1
            }
            known_genres |= new_known_genres

            # Update known combinations (fully resolved)
            new_known_combinations = {
                combo for combo in genre_combinations if len(set(combo) - known_genres) == 0
            }
            known_combinations |= new_known_combinations

            schedule.append(environment)
            iteration += 1

            logger.info(
                f"Environment {iteration}: {len(environment['known'])} known, "
                f"{len(environment['predict'])} predictable, "
                f"{len(environment['disjunct'])} disjunct"
            )

            if len(new_known_genres) > 0:
                logger.info(f"  ✓ New known genres: {sorted(new_known_genres)}")

            if len(environment['predict']) > 0:
                logger.info(f"  → Can predict these combinations: {environment['predict'][:5]}" +
                           (f" ... (+{len(environment['predict'])-5} more)" if len(environment['predict']) > 5 else ""))

            if len(environment['disjunct']) > 0:
                logger.info(f"  ✗ Still disjunct: {environment['disjunct'][:3]}" +
                           (f" ... (+{len(environment['disjunct'])-3} more)" if len(environment['disjunct']) > 3 else ""))

        # Check if we successfully resolved everything
        if len(schedule) > 0 and len(schedule[-1]["disjunct"]) > 0:
            unresolved = schedule[-1]["disjunct"]
            logger.error(
                f"Unable to bootstrap {len(unresolved)} genre combinations: {unresolved}"
            )
            logger.error(
                "These combinations have no overlap with single-genre treebanks."
            )

        return schedule

    def validate_schedule(self, schedule: List[Dict]) -> bool:
        """Check if schedule can resolve all combinations.

        Args:
            schedule: Bootstrap schedule

        Returns:
            True if all combinations can be resolved, False otherwise
        """
        if not schedule:
            return False

        # Last environment should have no disjunct combinations
        return len(schedule[-1]["disjunct"]) == 0

    def get_resolvable_genres(self, schedule: List[Dict]) -> Set[str]:
        """Get set of all genres that can be resolved.

        Args:
            schedule: Bootstrap schedule

        Returns:
            Set of genre labels
        """
        if not schedule:
            return set()

        # All known genres in final environment
        return set(schedule[-1]["known"])
