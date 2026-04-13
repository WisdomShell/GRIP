# coding=utf-8
import argparse
import sys
import json
from rouge_score import rouge_scorer
import utils


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate model predictions using EM, cover EM, F1 score, and ROUGE metrics.")
    parser.add_argument(
        "--references_path",
        default="/path/to/your/web.jsonl",
        help="Path to the JSONL file with reference answers."
    )
    parser.add_argument(
        "--predictions_path",
        default="/path/to/your/output/web.jsonl",
        help="Path to the JSONL file with model predictions."
    )
    parser.add_argument(
        "--answer_field",
        default="answer",
        choices=["answer", "answer_and_def_correct_predictions"],
        help="Reference field to use as ground-truth answers."
    )

    args = parser.parse_args()

    try:
        metrics = utils.evaluate_predictions(
            references_path=args.references_path,
            predictions_path=args.predictions_path,
            answer_field=args.answer_field
        )
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error during evaluation: {e}")
        sys.exit(1)

    ref_data = {}
    ref_raw_type = {}
    with open(args.references_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            question = item['question']
            raw = item[args.answer_field]
            ref_raw_type[question] = isinstance(raw, list)
            if isinstance(raw, list):
                answers = [str(a) for a in raw if a]
            else:
                answers = [str(raw)]
            ref_data[question] = answers

    gene_data = {}
    with open(args.predictions_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            question = item['question']
            prediction = item['prediction']
            if isinstance(prediction, list):
                prediction = prediction[-1] if len(prediction) > 0 else ""
            gene_data[question] = str(prediction).strip()

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []

    matched_count = 0
    skipped_count = 0

    for question in ref_data:
        if question not in gene_data:
            continue

        matched_count += 1
        raw_refs = ref_data[question]
        pred = gene_data[question].strip().lower()

        if not pred or pred == '0':
            skipped_count += 1
            continue

        references_lower = [r.strip().lower() for r in raw_refs]
        references_lower = [r for r in references_lower if r and r != '0']

        if not references_lower:
            skipped_count += 1
            continue

        if ref_raw_type[question]:
            matched_gt = None
            for ref in references_lower:
                if utils.em_score(pred, ref):
                    matched_gt = ref
                    break

            if matched_gt is not None:
                scores = scorer.score(matched_gt, pred)
                rouge1_scores.append(scores['rouge1'].fmeasure)
                rouge2_scores.append(scores['rouge2'].fmeasure)
                rougeL_scores.append(scores['rougeL'].fmeasure)
            else:
                best_r1, best_r2, best_rL = -1.0, -1.0, -1.0
                for ref in references_lower:
                    scores = scorer.score(ref, pred)
                    if scores['rouge1'].fmeasure > best_r1:
                        best_r1 = scores['rouge1'].fmeasure
                        best_r2 = scores['rouge2'].fmeasure
                        best_rL = scores['rougeL'].fmeasure
                rouge1_scores.append(best_r1)
                rouge2_scores.append(best_r2)
                rougeL_scores.append(best_rL)
        else:
            ref = references_lower[0]
            scores = scorer.score(ref, pred)
            rouge1_scores.append(scores['rouge1'].fmeasure)
            rouge2_scores.append(scores['rouge2'].fmeasure)
            rougeL_scores.append(scores['rougeL'].fmeasure)

    print(f"\n======== Data Statistic ========")
    if rouge1_scores:
        rouge1_avg = sum(rouge1_scores) / len(rouge1_scores)
        rouge2_avg = sum(rouge2_scores) / len(rouge2_scores)
        rougeL_avg = sum(rougeL_scores) / len(rougeL_scores)
        overall_avg = (rouge1_avg + rouge2_avg + rougeL_avg) / 3.0
    else:
        print("\nNot found valid data")
        overall_avg = 0.0

    print(f"Total questions: {metrics['num_total']}         Missing predictions: {metrics['missing_predictions']}")
    print("=" * 50)
    print(f"Exact Match Score:")
    print(f"EM: {metrics['em_accuracy']:.4f} ({metrics['em_correct']}/{metrics['num_total']})")
    print()
    print(f"F1 Score:")
    print(f"F1: {metrics['avg_f1']:.4f}")
    print()
    print(f"ROUGE Score:")
    print(f"ROUGE: {overall_avg:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()