"""Original AUG-PE mechanism from the ICLR 2024 paper.

This wraps the existing logic from src/main.py into the Mechanism interface.
The algorithmic steps are preserved: DP NN histogram -> rank/prob selection ->
variation expansion.
"""

import collections
import logging
import os
from typing import Any, Dict, List, Tuple

import numpy as np

from ..registry import register
from .base import Mechanism, RoundResult

# Heavy imports are deferred to method bodies to allow lightweight import
# of the module for testing without GPU/faiss/sentence_transformers.


@register("mechanism", "original_aug_pe")
class OriginalAugPEMechanism(Mechanism):
    """Original AUG-PE mechanism (Algorithm 1 from the paper).

    Selection modes:
    - "rank": sort by noisy counts descending, take top N (default in scripts)
    - "prob": sample with replacement proportional to noisy counts
    """

    def generate_initial(
        self,
        num_samples: int,
        label_counter: collections.Counter,
    ) -> Tuple[List[str], List[str], collections.Counter]:
        """Generate initial synthetic pool via RANDOM_API."""
        num_seed = int(num_samples / self.init_combine_divide_L)

        samples, labels, sync_counter, prefix_prompts = self.api.text_random_sampling(
            num_samples=num_seed,
            prompt_counter=label_counter,
            lens_dict=None,
        )
        return samples, labels, sync_counter

    def compute_weights(self, noisy_counts: np.ndarray, t: int) -> np.ndarray:
        """Original mechanism: return noisy counts as-is.

        The selection logic (rank vs prob) is handled in _select_survivors.
        """
        return noisy_counts

    def _select_survivors(
        self,
        counts: np.ndarray,
        current_idx: int,
        selected_size: int,
    ) -> List[int]:
        """Select survivor indices based on DP histogram counts.

        Args:
            counts: noisy count vector for one class.
            current_idx: offset into the global sample array.
            selected_size: number of survivors to select.

        Returns:
            List of global indices of selected survivors.
        """
        if self.select_syn_mode == 'prob':
            candidate_indices = np.arange(
                current_idx, current_idx + len(counts), dtype=int)
            sampling_prob = counts / np.sum(counts)
            sub_new_indices = np.random.choice(
                candidate_indices, size=selected_size, p=sampling_prob)
            return sub_new_indices.tolist()

        elif self.select_syn_mode == 'rank':
            sort_index = [
                i + current_idx
                for i, x in sorted(enumerate(counts), key=lambda x: -x[1])
            ]
            return sort_index[:selected_size]
        else:
            raise ValueError(f"Unsupported select_syn_mode: {self.select_syn_mode}")

    def run_round(
        self,
        t: int,
        syn_samples: List[str],
        additional_info: List[str],
        sync_labels_counter: collections.Counter,
        private_classes: List[str],
        all_private_features: np.ndarray,
        private_labels_indexer: Dict[str, List[int]],
        result_folder: str,
    ) -> RoundResult:
        """Execute one PE round: embed -> count -> select -> vary."""
        from dpsda.dp_counter import dp_nn_histogram
        from dpsda.feature_extractor import extract_features
        from dpsda.logging import log_count, log_num_words, log_prompt_generation

        logging.info(f"Round t={t}")

        # Step 1: Build synthetic embeddings E_t
        if self.lookahead_degree == 0:
            packed_samples = np.expand_dims(syn_samples, axis=1)
        else:
            logging.info("Running lookahead text variation")
            packed_samples, _, _, all_gen_words, all_masked_prompts = self.api.text_variation(
                sequences=syn_samples,
                additional_info=additional_info,
                num_variations_per_sequence=self.lookahead_degree,
                variation_degree=self.variation_degree_schedule[t],
            )
            if self.lookahead_self:
                packed_samples = np.concatenate(
                    (packed_samples, np.expand_dims(syn_samples, axis=1)), axis=1)

            os.makedirs(f'{result_folder}/{t}', exist_ok=True)
            log_num_words(
                fname=f'{result_folder}/{t}/num_word_lookahead.csv',
                all_gen_words=all_gen_words, all_target_words=[])
            log_prompt_generation(
                fname=f'{result_folder}/{t}/prompt_generation.jsonl',
                prompts=all_masked_prompts, generations=packed_samples)

        # Step 2: Extract features for each lookahead variation, then average
        packed_features = []
        logging.info("Running feature extraction")
        for i in range(packed_samples.shape[1]):
            sub_features = extract_features(
                data=packed_samples[:, i],
                batch_size=self.feature_extractor_batch_size,
                model_name=self.feature_extractor,
            )
            packed_features.append(sub_features)
        packed_features = np.mean(packed_features, axis=0)
        logging.info(f"Feature extraction shape {packed_features.shape}")

        # Step 3: Per-class DP histogram, selection, and variation
        logging.info("Computing histogram and selecting survivors")
        count_per_class = {}
        new_syn_samples = []
        new_additional_info = []
        all_selected_samples = []
        all_selected_additional_info = []
        current_idx = 0

        for class_ in private_classes:
            num_samples_per_class = sync_labels_counter[class_]
            if num_samples_per_class == 0:
                continue

            public_features = packed_features[current_idx:current_idx + num_samples_per_class]
            assert num_samples_per_class == public_features.shape[0]

            selected_size = int(num_samples_per_class / self.combine_divide_L)
            logging.info(f"{class_}, n={num_samples_per_class}, selected_size={selected_size}")

            if selected_size == 0:
                # Edge case: too few samples to select from
                sub_new_indices = list(range(current_idx, current_idx + num_samples_per_class))
                selected_syn = [syn_samples[i] for i in sub_new_indices]
                selected_info = [additional_info[i] for i in sub_new_indices]
                new_variants_samples = selected_syn * self.combine_divide_L
                new_variants_info = selected_info * self.combine_divide_L
                count_per_class[class_] = (np.array([]), np.array([]))
            else:
                # DP nearest-neighbor histogram
                sub_count, sub_clean_count = dp_nn_histogram(
                    public_features=public_features,
                    private_features=all_private_features[private_labels_indexer[class_]],
                    noise_multiplier=self.noise_multiplier,
                    num_nearest_neighbor=self.num_nearest_neighbor,
                    mode=self.nn_mode,
                    threshold=self.count_threshold,
                )
                assert np.sum(sub_count) > 0

                count_per_class[class_] = (sub_count, sub_clean_count)

                # Apply weight computation (identity for original mechanism)
                weights = self.compute_weights(sub_count, t)

                # Select survivors
                sub_new_indices = self._select_survivors(weights, current_idx, selected_size)

                # Log counts
                count_fname = class_.replace("\t", "_").replace(
                    " ", "_").replace("&", "").replace(":", "")
                log_count(sub_count, sub_clean_count,
                          f'{result_folder}/{t}/count_class/{count_fname}.csv')

                selected_syn = [syn_samples[i] for i in sub_new_indices]
                selected_info = [additional_info[i] for i in sub_new_indices]
                assert len(selected_syn) == len(selected_info)

                # Generate variations for next round
                new_variants_samples = []
                if self.combine_divide_L == 1:
                    num_vars = 1
                elif self.combine_divide_L > 1:
                    if self.donnot_keep_last_iter:
                        num_vars = self.combine_divide_L
                    else:
                        num_vars = self.combine_divide_L - 1
                        new_variants_samples.extend(selected_syn)
                else:
                    raise ValueError("combine_divide_L must be >= 1")

                logging.info(f"num_variations_per_sequence={num_vars}")

                new_variants_stacked, _, _, _, _ = self.api.text_variation(
                    sequences=selected_syn,
                    additional_info=selected_info,
                    num_variations_per_sequence=num_vars,
                    variation_degree=self.variation_degree_schedule[t],
                )
                for x in new_variants_stacked:
                    new_variants_samples.extend(x.tolist())

                new_variants_info = selected_info * self.combine_divide_L

            new_syn_samples.extend(new_variants_samples)
            new_additional_info.extend(new_variants_info)
            sync_labels_counter[class_] = len(new_variants_samples)

            # Accumulate for saving based on save mode
            if self.save_syn_mode == 'selected':
                all_selected_samples.extend(selected_syn)
                all_selected_additional_info.extend(selected_info)
            elif self.save_syn_mode == 'one_var':
                all_selected_samples.extend(new_variants_stacked[:, 0])
                all_selected_additional_info.extend(selected_info)
            elif self.save_syn_mode == 'all':
                all_selected_samples.extend(new_variants_samples)
                all_selected_additional_info.extend(new_variants_info)

            current_idx += public_features.shape[0]

        return RoundResult(
            selected_samples=all_selected_samples,
            selected_labels=all_selected_additional_info,
            next_round_samples=new_syn_samples,
            next_round_labels=new_additional_info,
            updated_label_counter=sync_labels_counter,
            count_per_class=count_per_class,
        )
