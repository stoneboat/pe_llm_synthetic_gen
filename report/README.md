# Yelp DP Experiment Report

This folder collects the key outputs for the two Yelp experiment variants:

- `pe_top_k_dp_eps1/`
- `original_dp_eps1/`

Each subfolder contains:

- `all_results.json`: downstream classification summary for iteration 10
- `fid.csv`: FID values across iterations

## What to expect

When comparing the two methods, the main signals are:

- Higher classification accuracy is better.
- Lower FID is better.

For `all_results.json`, the most useful fields are:

- `test_accuracy_all`: overall test accuracy
- `eval_accuracy_all`: validation accuracy
- `train_loss`: training loss

For `fid.csv`, each row starts with the iteration number followed by the FID for that iteration.

## How to read the comparison

In a strong result, we would like to see:

- test accuracy stay the same or increase
- FID stay the same or decrease

In practice, there is usually a tradeoff:

- one method may have better downstream accuracy
- the other may have slightly better FID

So the expected interpretation is:

- if a method has higher `test_accuracy_all`, it is better for the downstream classifier
- if a method has lower FID, it is better at matching the real data distribution
- if one wins on accuracy and the other wins on FID, the choice depends on which objective matters more for the experiment
