from typing import List, Dict

def rank_hypotheses(hypotheses: List[Dict], weights: Dict = None) -> List[Dict]:
    if weights is None:
        weights = {"novelty": 0.3, "impact": 0.4, "risk": 0.3}
    
    scored = []
    for hyp in hypotheses:
        novelty = float(hyp.get('novelty', 0.5)) if isinstance(hyp.get('novelty'), (int, float)) else 0.5
        impact = float(hyp.get('impact', 0.5)) if isinstance(hyp.get('impact'), (int, float)) else 0.5
        risk = float(hyp.get('risk', 0.5)) if isinstance(hyp.get('risk'), (int, float)) else 0.5
        
        risk_score = 1 - risk
        total = (weights['novelty'] * novelty +
                 weights['impact'] * impact +
                 weights['risk'] * risk_score)
        hyp['score'] = round(total, 3)
        scored.append(hyp)
    
    scored.sort(key=lambda x: x.get('score', 0), reverse=True)
    return scored