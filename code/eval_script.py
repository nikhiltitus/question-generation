from nlgeval import compute_metrics
metrics_dict = compute_metrics(hypothesis='candidate.txt',references=['reference.txt'])
