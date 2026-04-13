import json
import re
import string
import collections


def normalize_answer(s):
    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def cover_em_score(prediction: str, ground_truth: str) -> bool:
    normalized_pred = normalize_answer(prediction)
    normalized_gt = normalize_answer(ground_truth)
    return normalized_gt in normalized_pred


def em_score(prediction: str, ground_truth: str) -> bool:
    normalized_pred = normalize_answer(prediction)
    normalized_gt = normalize_answer(ground_truth)
    return normalized_pred == normalized_gt


def get_tokens(s):
    if not s:
        return []
    return normalize_answer(s).split()


def f1_score(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def evaluate_predictions_impl(references: dict, predictions: dict) -> dict:
    missing_predictions = 0
    em_correct = 0
    cover_em_correct = 0
    total_f1 = 0.0

    for question, answers in references.items():
        if question not in predictions:
            missing_predictions += 1
            continue

        pred = predictions[question]

        if isinstance(pred, list):
            pred = pred[-1] if len(pred) > 0 else ""
        pred = str(pred).strip()

        if not answers:
            total_f1 += 0.0
            continue

        gt_for_f1 = str(answers[0]).strip()
        if pred and pred != '0':
            best_f1 = f1_score(gt_for_f1, pred)
        else:
            best_f1 = 0.0

        best_em = False
        best_cover_em = False

        if pred and pred != '0':
            for gt in answers:
                if em_score(pred, gt):
                    best_em = True
                if cover_em_score(pred, gt):
                    best_cover_em = True

        em_correct += int(best_em)
        cover_em_correct += int(best_cover_em)
        total_f1 += best_f1

    total = len(references) - missing_predictions
    em_accuracy = em_correct / total if total > 0 else 0.0
    cover_em_accuracy = cover_em_correct / total if total > 0 else 0.0
    avg_f1 = total_f1 / total if total > 0 else 0.0

    return {
        'missing_predictions': missing_predictions,
        'num_total': total,
        'em_correct': em_correct,
        'em_accuracy': em_accuracy,
        'cover_em_correct': cover_em_correct,
        'cover_em_accuracy': cover_em_accuracy,
        'avg_f1': avg_f1,
        'num_correct': em_correct,
        'accuracy': em_accuracy,
    }


def evaluate_predictions(
    references_path: str,
    predictions_path: str,
    answer_field: str = "answer"
) -> dict:
    references = {}
    with open(references_path, 'r', encoding='utf-8') as ref_file:
        for line in ref_file:
            example = json.loads(line)
            q = example.get('question')
            answers = example.get(answer_field)
            if isinstance(answers, str):
                answers = [answers]
            elif answers is None:
                answers = []
            references[q] = answers

    predictions = {}
    with open(predictions_path, 'r', encoding='utf-8') as pred_file:
        for line in pred_file:
            example = json.loads(line)
            predictions[example.get('question')] = example.get('prediction')

    return evaluate_predictions_impl(
        references=references,
        predictions=predictions
    )