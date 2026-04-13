from verl.utils.reward_score.grip import compute_score_grip

def compute_scores(
    data_source: str,
    solution_str: str, 
    ground_truth: str,
    extra_info=None,
    tokenizer=None,
) -> float:
    if extra_info['data_source'] == 'GRIPRL':
        return compute_score_grip(data_source, solution_str, ground_truth, extra_info, tokenizer)
    else:
        raise ValueError(f"Unknown data source: {extra_info}")