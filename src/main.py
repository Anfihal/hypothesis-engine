import os
import sys
import json
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.nlp.extractor import EntityExtractor
from src.graph.kgraph import KnowledgeGraph
from src.generation.deliberation import HypothesisDeliberation
from src.ranking.scorer import rank_hypotheses

load_dotenv()

def main():
    kpi = "Increase creep resistance of nickel-based superalloy by 20% at 700°C"
    constraints = "No rhenium, budget < $500k, scalable to industrial production"
    language = "ru"  # можно менять: 'en', 'ru', 'zh'

    literature = """
    The addition of 0.5% niobium to Inconel 718 significantly increases yield strength at elevated temperatures.
    Previous studies show that niobium forms stable carbides which hinder dislocation motion.
    However, excess niobium may cause embrittlement due to phase precipitation.
    Recent experiments indicate that a two-step aging treatment can further enhance creep resistance.
    Also, the use of tantalum has been explored but is expensive and limited in supply.
    Grain boundary engineering through thermomechanical processing improves durability.
    """

    extractor = EntityExtractor()
    analysis = extractor.process_document(literature)

    kg = KnowledgeGraph()
    for ent in analysis['entities']:
        kg.add_entity(ent['text'], ent['label'])
    for rel in analysis['relationships']:
        kg.add_relation(rel['subject'], rel['relation'], rel['object'], rel['evidence'])

    graph_context = kg.to_text_context()
    print("=== Knowledge Graph ===")
    print(graph_context)
    print("\n")

    delib = HypothesisDeliberation(use_yandex=False, model_name="llama3.1:8b", timeout=45)
    full_context = graph_context + "\nLiterature insights: Niobium forms carbides; two-step aging improves properties."

    print(f"Starting generation (language: {language})...")
    hypotheses = delib.run(kpi, constraints, full_context, language=language, max_rounds=2)

    # Ранжирование
    for hyp in hypotheses:
        if 'novelty' not in hyp:
            hyp['novelty'] = 0.7
        if 'impact' not in hyp:
            hyp['impact'] = 0.6
        if 'risk' not in hyp:
            hyp['risk'] = 0.3
        if isinstance(hyp.get('risk'), str):
            risk_map = {'low': 0.2, 'medium': 0.5, 'high': 0.8}
            hyp['risk'] = risk_map.get(hyp['risk'].lower(), 0.5)
        if isinstance(hyp.get('impact'), str):
            import re
            nums = re.findall(r'\d+', hyp['impact'])
            if nums:
                hyp['impact'] = float(nums[0]) / 100.0
            else:
                hyp['impact'] = 0.6
        if isinstance(hyp.get('novelty'), str):
            hyp['novelty'] = 0.7 if 'novel' in hyp['novelty'].lower() else 0.5

    ranked = rank_hypotheses(hypotheses)

    print("\n=== RANKED HYPOTHESES ===\n")
    for i, hyp in enumerate(ranked, 1):
        print(f"--- Hypothesis {i} (Score: {hyp.get('score', 'N/A')}) ---")
        print(f"Statement: {hyp.get('statement', 'N/A')}")
        print(f"Mechanism: {hyp.get('mechanism', 'N/A')}")
        print(f"Novelty: {hyp.get('novelty', 'N/A')}")
        print(f"Risk: {hyp.get('risk', 'N/A')}")
        print(f"Impact: {hyp.get('impact', 'N/A')}")
        print(f"Sources: {', '.join(hyp.get('sources', ['не указаны']))}")
        print(f"Explanation: {hyp.get('explanation', 'N/A')}")
        print(f"Recommendation: {hyp.get('recommendation', 'N/A')}")
        print(f"Generation time: {hyp.get('generation_time', 'N/A')} sec\n")

    output = {
        "kpi": kpi,
        "constraints": constraints,
        "language": language,
        "hypotheses": ranked
    }
    with open("hypotheses_output.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print("Результаты сохранены в hypotheses_output.json")

if __name__ == "__main__":
    main()