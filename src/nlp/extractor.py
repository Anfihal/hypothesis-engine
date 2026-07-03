import re
from typing import List, Dict, Any

class EntityExtractor:
    def __init__(self):
        # Словарь для материалов и элементов (можно расширять)
        self.materials = {
            'inconel', 'steel', 'alloy', 'superalloy', 'niobium', 'ni', 'nb',
            'chromium', 'cr', 'iron', 'fe', 'nickel', 'ni', 'titanium', 'ti',
            'aluminium', 'al', 'molybdenum', 'mo', 'tungsten', 'w', 'vanadium', 'v',
            'cobalt', 'co', 'copper', 'cu', 'manganese', 'mn', 'silicon', 'si',
            'carbon', 'c', 'boron', 'b', 'zirconium', 'zr', 'hafnium', 'hf',
            'tantalum', 'ta', 'rhenium', 're', 'iridium', 'ir', 'platinum', 'pt',
            'palladium', 'pd', 'rhodium', 'rh', 'ruthenium', 'ru', 'osmium', 'os'
        }
        self.patterns = {
            'material': r'\b(' + '|'.join(self.materials) + r')\b',
            'percent': r'\b\d+\.?\d*\s*%',
            'temperature': r'\b\d+\s*°C\b|\b\d+\s*degrees\b',
            'time': r'\b\d+\s*(min|hour|h|day)\b',
            'property': r'\b(yield strength|creep resistance|hardness|toughness|ductility|elastic modulus|conductivity|thermal expansion)\b',
        }
        self.relation_patterns = [
            (r'(\w+)\s+(increases|enhances|improves|boosts)\s+(\w+)', 'enhances'),
            (r'(\w+)\s+(decreases|reduces|lowers)\s+(\w+)', 'reduces'),
            (r'(\w+)\s+is added to\s+(\w+)', 'added_to'),
            (r'(\w+)\s+forms\s+(\w+)', 'forms'),
            (r'(\w+)\s+depends on\s+(\w+)', 'depends_on'),
            (r'(\w+)\s+inhibits\s+(\w+)', 'inhibits'),
        ]
    
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        entities = []
        for label, pattern in self.patterns.items():
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entities.append({
                    'text': match.group(0),
                    'label': label,
                    'start': match.start(),
                    'end': match.end()
                })
        return entities
    
    def extract_relationships(self, text: str) -> List[Dict[str, str]]:
        relationships = []
        for pattern, rel_type in self.relation_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                subj, obj = match.group(1), match.group(2)
                relationships.append({
                    'subject': subj.strip(),
                    'relation': rel_type,
                    'object': obj.strip(),
                    'evidence': match.group(0)
                })
        return relationships
    
    def process_document(self, text: str) -> Dict:
        return {
            'entities': self.extract_entities(text),
            'relationships': self.extract_relationships(text)
        }