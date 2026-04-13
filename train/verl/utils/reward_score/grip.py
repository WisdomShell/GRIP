import re
import json
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def score_prediction(pred, gold):
    """
    评分函数，根据预测结果的格式和内容给出分数
    
    Args:
        pred (str): 预测结果字符串
        gold (str): 标准答案字符串
    
    Returns:
        float: 总分数 (FormatScore + ScoreAnswer)
    """
    
    # 初始化分数
    format_score = 0
    score_answer = 0
    
    def parse_format(text):
        """解析文本格式并提取相应内容"""
        # Case 1: [ANSWER] Complete answer [SOLVED]
        case1_pattern = r'\[ANSWER\]\s*(.*?)\s*\[SOLVED\]'
        case1_match = re.search(case1_pattern, text, re.DOTALL | re.IGNORECASE)
        
        # Case 2: [INTERMEDIARY] Partial answer [RETRIEVE] New Query.
        case2_pattern = r'\[Intermediary\]\s*(.*?)\s*\[RETRIEVE\]\s*(.*?)$'
        case2_match = re.search(case2_pattern, text, re.DOTALL | re.IGNORECASE)
        
        if case1_match:
            return 'case1', {'complete_answer': case1_match.group(1).strip()}
        elif case2_match:
            return 'case2', {
                'partial_answer': case2_match.group(1).strip(),
                'new_query': case2_match.group(2).strip()
            }
        else:
            return 'unknown', {}
    
    def calculate_bleu(reference, candidate):
        """计算BLEU分数"""
        smoothing = SmoothingFunction()
        
        if not reference or not candidate:
            return 0.0
        
        ref_words = reference.lower().split()
        cand_words = candidate.lower().split()
        if not ref_words or not cand_words:
            return 0.0
            
        return sentence_bleu([ref_words], cand_words, weights=(1, 0, 0, 0), smoothing_function=smoothing.method1)
    
    # 解析预测结果和标准答案的格式
    pred_format, pred_content = parse_format(pred)
    gold_format, gold_content = parse_format(gold)
    
    # 计算格式分数
    if pred_format == gold_format and pred_format != 'unknown':
        format_score = 0.5
    else:
        format_score = 0.0
    
    # 计算内容分数
    if format_score > 0:  # 格式一致
        if pred_format == 'case1':
            # Case 1: 比较 Complete answer
            score_answer = calculate_bleu(
                gold_content.get('complete_answer', ''),
                pred_content.get('complete_answer', '')
            )
        elif pred_format == 'case2':
            # Case 2: 分别比较 Partial answer 和 New Query，取平均
            partial_bleu = calculate_bleu(
                gold_content.get('partial_answer', ''),
                pred_content.get('partial_answer', '')
            )
            query_bleu = calculate_bleu(
                gold_content.get('new_query', ''),
                pred_content.get('new_query', '')
            )
            score_answer = (partial_bleu + query_bleu) / 2
    else:  # 格式不一致
        # 比较 Complete answer 和 Partial answer
        gold_text = ''
        pred_text = ''
        
        if gold_format == 'case1':
            gold_text = gold_content.get('complete_answer', '')
        elif gold_format == 'case2':
            gold_text = gold_content.get('partial_answer', '')
        
        if pred_format == 'case1':
            pred_text = pred_content.get('complete_answer', '')
        elif pred_format == 'case2':
            pred_text = pred_content.get('partial_answer', '')

        score_answer = calculate_bleu(gold_text, pred_text)
    
    # 返回总分数
    total_score = format_score + score_answer
    return total_score

def compute_score_grip(
    data_source: str,
    solution_str: str, 
    ground_truth: str,
    extra_info=None,
    tokenizer=None,    
) -> float:
    gt = str(json.loads(ground_truth)['label']).strip()
    return score_prediction(solution_str, gt)