"""Compare port with original AST-extracted function, plus temporal leakage check."""
import ast
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from researchforge_scoring import original_score, historical_topics


class OriginalParity(unittest.TestCase):
    def test_original_function_parity(self):
        source = Path(__file__).resolve().parents[3]/'work/ResearchForge-AI/agent.py'
        tree = ast.parse(source.read_text(encoding='utf-8'))
        node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'calculate_advanced_match')
        # Only the scorer body is executed, with inert model/output containers.
        node.returns = None
        for arg in node.args.args:
            arg.annotation = None
        env = dict(np=np, cosine_similarity=cosine_similarity,
                   ProjectRequirement=type('ProjectRequirement', (), {}), MatchResult=lambda **kwargs:NS(**kwargs))
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])), str(source), 'exec'), env)
        dummy = NS(embedding_model=True, _generate_explanation=lambda *args:'unused')
        rng = np.random.default_rng(42)
        for i in range(25):
            skills = ['Python', 'ML'][:i % 3]
            topics = ['cs.AI', 'math'][:i % 3]
            embedding = rng.normal(size=5).tolist() if i % 2 else None
            project_embedding = rng.normal(size=5).tolist() if i % 2 else None
            kind = [None, 'remote', 'International'][i % 3]
            r = NS(id='r', name='r', skills=skills, research_interests=[NS(topic=t,years_experience=3) for t in topics], embedding=embedding,citation_count=0)
            p = NS(id='p',title='p',required_skills=['python'],research_areas=['cs'],collaboration_type=kind,difficulty_level='Intermediate',embedding=project_embedding)
            expected = env['calculate_advanced_match'](dummy,r,p).overall_score
            cosine = float(cosine_similarity([embedding],[project_embedding])[0,0]) if embedding else None
            self.assertEqual(original_score(skills,['python'],topics,['cs'],kind,cosine),expected)

    def test_no_same_day_or_future_categories(self):
        papers = {'old':dict(categories=['cs.AI']), 'same':dict(categories=['leak']), 'future':dict(categories=['leak2'])}
        self.assertEqual(historical_topics('a',10,papers,{'a':['old','same','future']},{'a':[9,10,11]}),['cs.AI'])


if __name__ == '__main__':
    unittest.main()
