import pandas as pd
import yaml
from typing import Dict, List, Any

class TestConfigGenerator:
    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path)
        
    def create_test_config(self) -> Dict[str, Any]:
        return {
            "description": "My model port 8085",
            "prompts": ["{{question}}"],
            "providers": [
                {
                    "id": "http",
                    "config": {
                        "request": """ |-
                            POST /cosmetics-answer HTTP/1.1
                            Host: 127.0.0.1:8085
                            Content-Type: application/json
                            Accept: application/json
                            
                            {"question": "{{question}}"}
                        """,
                        "transformResponse": "json.answer"
                    }
                }
            ],
            "tests": self.generate_tests(),
            "sharing": {
                "appBaseUrl": "https://promptfoo.imutably.com"
            }
        }
    
    def generate_tests(self) -> List[Dict[str, Any]]:
        tests = []
        
        for _, row in self.df.iterrows():
            test_case = {
                "vars": {
                    "question": row['Question']
                },
                "assert": []
            }
            
            expected = str(row['Answer']).strip()
            test_case["assert"].append({
                "type": "llm-rubric",
                "value": expected
            })
            
            tests.append(test_case)
            
        return tests
    
    def save_yaml(self, output_path: str):
        config = self.create_test_config()
        
        with open(output_path, 'w', encoding='utf-8') as file:
            yaml.dump(config, file, allow_unicode=True, sort_keys=False, default_flow_style=False)
        
        print(f"Test configuration saved to {output_path}")


csv_path = r"data_test/LF1/LF-1.csv"
yaml_path = "promptfooconfig.yaml"

# Create and save the test configuration
generator = TestConfigGenerator(csv_path)
generator.save_yaml(yaml_path)
