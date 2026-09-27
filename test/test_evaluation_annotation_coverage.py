import unittest

from src.evaluation.benchmark import fingerprint, score


class AnnotationCitationCoverageTest(unittest.TestCase):
    def setUp(self):
        self.citation = '[1512.03385:h0004-dcb100dff33d:s0002-ad1e3a167a8c]'
        self.other = '[1512.03385:h0007-64d24f81eb5a:s0002-9ebf348eb67f]'
        self.response = f'第一项事实 {self.citation}。第二项事实 {self.citation}。'
        self.cases = [{'id': 'case-1', 'question': '测试问题'}]
        self.run = {
            'dataset_hash': fingerprint(self.cases),
            'profile': {'model': 'test-model', 'prompt_version': 'test-prompt'},
            'predictions': [{'case_id': 'case-1', 'response': self.response, 'status': 'completed'}],
        }

    def annotation(self, citation_ids):
        return [{
            'case_id': 'case-1', 'response_hash': fingerprint(self.response), 'reviewer': 'human',
            'claims': [{'text': '第一项事实', 'supported': True, 'citation_supported': True}],
            'citations': [{'text': c, 'supported': True} for c in citation_ids],
        }]

    def test_repeated_citation_must_be_judged_for_each_occurrence(self):
        with self.assertRaisesRegex(ValueError, '逐次覆盖'):
            score(self.cases, self.run, annotations=self.annotation([self.citation]))

    def test_unmentioned_citation_cannot_replace_response_citation(self):
        with self.assertRaisesRegex(ValueError, '逐次覆盖'):
            score(self.cases, self.run, annotations=self.annotation([self.citation, self.other]))

    def test_citation_cannot_be_labeled_when_response_has_none(self):
        self.run['predictions'][0]['response'] = '没有引用的回答。'
        annotation = self.annotation([self.citation])
        annotation[0]['response_hash'] = fingerprint(self.run['predictions'][0]['response'])
        with self.assertRaisesRegex(ValueError, '逐次覆盖'):
            score(self.cases, self.run, annotations=annotation)

    def test_complete_occurrence_coverage_scores(self):
        result = score(self.cases, self.run, annotations=self.annotation([self.citation, self.citation]))
        self.assertEqual(result['summary']['citation_precision']['count'], 1)
        self.assertEqual(result['summary']['citation_precision']['mean'], 1)


if __name__ == '__main__':
    unittest.main()
