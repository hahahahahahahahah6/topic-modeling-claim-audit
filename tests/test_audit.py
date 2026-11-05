import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from topic_modeling_claim_audit.audit import audit, has_evidence, main, match_claims
from topic_modeling_claim_audit.phrases import TOPIC, NOTOPIC, GENERIC_NAMES

class AuditTests(unittest.TestCase):
    def test_phrase_oracle(self):
        self.assertEqual(TOPIC, ['topic modeling', 'topic modelling', 'topic model', 'topic models', 'topic', 'topics', 'discovers topics', 'discover topics', 'extracts topics', 'extract topics', 'topic modeling support', 'supports topic modeling', 'provides topic modeling', 'performs topic modeling', 'topic modeling feature', 'topic modeling capability', 'topic modeling system', 'topic modeling pipeline', 'topic modeling step', 'topic modeling process', 'topic modeling model', 'topic modeler', 'topic modelling support', 'supports topic modelling', 'topic extraction', 'extracts topic', 'extract topic', 'topic detection', 'detects topics', 'detect topics', 'topic identification', 'identifies topics', 'identify topics', 'topic discovery', 'discovers topic', 'top topics', 'topic ranking', 'ranked topics', 'topic scoring', 'automatic topic modeling', 'auto topic modeling', 'latent topic modeling', 'latent topics', 'topic distribution', 'topic assignments', 'assigns topics', 'assign topics', 'document topic modeling', 'text topic modeling', 'topic analysis'])
        self.assertEqual(NOTOPIC, ['does not model topics', 'do not model topics', "doesn't model topics", "don't model topics", 'no topic modeling', 'without topic modeling', 'never models topics', 'does not perform topic modeling', 'never performs topic modeling', 'topic modeling not supported', 'topic modeling unsupported', 'no topic modeling support', 'lacks topic modeling', 'missing topic modeling', 'topic modeling disabled', 'no topic model', 'without a topic model', 'does not provide topic modeling', 'does not support topic modeling', 'no automatic topic modeling', 'without topic modeling support', 'does not extract topics', 'topic modeling not available', 'no topic modelling'])

    def test_generic_oracle(self):
        self.assertEqual(GENERIC_NAMES, ['extract_topics', 'extract_topic', 'get_topics', 'get_topic', 'discover_topics', 'discover_topic', 'detect_topics', 'detect_topic', 'find_topics', 'find_topic', 'identify_topics', 'identify_topic', 'model_topics', 'fit_topic_model', 'get_topic_model', 'topic_modeling', 'topic_modelling', 'assign_topics', 'get_document_topics', 'show_topics', 'print_topics', 'get_topics_distribution', 'topic_distribution', 'infer_topics'])

    def test_positive_pairs(self):
        routes = [
            ('gensim.models', 'LdaModel', ['get_topics', 'show_topics', 'print_topics', 'get_document_topics']),
            ('gensim.models', 'LdaMulticore', ['get_topics', 'show_topics', 'print_topics', 'get_document_topics']),
            ('sklearn.decomposition', 'LatentDirichletAllocation', ['fit', 'fit_transform']),
            ('sklearn.decomposition', 'NMF', ['fit', 'fit_transform']),
            ('bertopic', 'BERTopic', ['fit_transform', 'fit']),
            ('gensim.models.wrappers', 'LdaMallet', ['show_topics', 'get_topics', 'print_topics']),
            ('gensim.models', 'Nmf', ['get_topics', 'show_topics']),
        ]
        for module, cls, methods in routes:
            for method in methods:
                for imp, ctor in [(f'from {module} import {cls}', cls), (f'from {module} import {cls} as Model', 'Model'), (f'import {module} as lib', 'lib.' + cls), (f'import {module}', module + '.' + cls)]:
                    for body in [f'm = {ctor}()\nm.{method}(data)', f'{ctor}().{method}(data)']:
                        with self.subTest(source=imp + body):
                            self.assertTrue(has_evidence(imp + '\n' + body))

    def test_generic_calls(self):
        for name in GENERIC_NAMES:
            for source in [f'x.{name}()', f'from arbitrary import {name}\n{name}()', f'from arbitrary import {name} as f\nf()']:
                with self.subTest(source=source):
                    self.assertTrue(has_evidence(source))

    def test_rejections(self):
        cases = [
            'from rake_nltk import Rake\nr=Rake()\nr.extract_keywords_from_text(text)',
            'from yake import KeywordExtractor\nk=KeywordExtractor()\nk.extract_keywords(text)',
            'from keybert import KeyBERT\nk=KeyBERT()\nk.extract_keywords(text)',
            'from sklearn.feature_extraction.text import TfidfVectorizer\nv=TfidfVectorizer()\nx=v.fit_transform(docs)\nv.get_feature_names_out()\nx.toarray()',
            'import gensim.summarization\ngensim.summarization.keywords(text)',
            'from gensim.summarization import summarize\nsummarize(text)',
            'from sumy.summarizers.lsa import LsaSummarizer\ns=LsaSummarizer()\ns.summarize(text)',
            'from transformers import pipeline\npipeline("summarization")',
            'from gensim.models import Word2Vec\nm=Word2Vec()\nm.wv',
            'from gensim.models import Doc2Vec\nm=Doc2Vec()\nm.wv',
            'from transformers import pipeline\npipeline("sentiment-analysis")',
            'nlp.ner(text)', 'nlp.pos_tag(text)', 'text.split()', 'stemmer.stem(text)', 'lemmatizer.lemmatize(text)',
            '"topic topics"\n# topic\nvalue = "topics"',
            'import subprocess\nsubprocess.run(["external-topic-cli"])',
            'extract_topics()', 'extract_topics',
        ]
        for cls in ['KMeans', 'DBSCAN', 'AgglomerativeClustering']:
            for method in ['fit', 'fit_predict']:
                cases.append(f'from sklearn.cluster import {cls}\nm={cls}()\nm.{method}(x)')
        for method in ['fit', 'fit_transform']:
            cases.append(f'from sklearn.decomposition import TruncatedSVD\nm=TruncatedSVD()\nm.{method}(x)')
        for module, cls in [('gensim.models','LdaModel'), ('gensim.models','LdaMulticore'), ('sklearn.decomposition','LatentDirichletAllocation'), ('sklearn.decomposition','NMF'), ('bertopic','BERTopic'), ('gensim.models.wrappers','LdaMallet'), ('gensim.models','Nmf')]:
            cases.extend([f'from {module} import {cls}', f'from {module} import {cls}\nm={cls}()', f'from {module} import {cls}\n{cls}'])
        for name in GENERIC_NAMES:
            cases.extend([f'from arbitrary import {name}', f'from arbitrary import {name}\n{name}'])
        for source in cases:
            with self.subTest(source=source):
                self.assertFalse(has_evidence(source))

    def test_shadowing(self):
        for name in GENERIC_NAMES:
            self.assertFalse(has_evidence(f'from lib import {name}\ndef {name}(): pass\n{name}()'))
        for module, cls, method in [('gensim.models','LdaModel','get_topics'), ('gensim.models','LdaMulticore','show_topics'), ('sklearn.decomposition','LatentDirichletAllocation','fit'), ('sklearn.decomposition','NMF','fit_transform'), ('bertopic','BERTopic','fit'), ('gensim.models.wrappers','LdaMallet','show_topics'), ('gensim.models','Nmf','get_topics')]:
            for form in [f'm={cls}()\nm.{method}()', f'{cls}().{method}()']:
                with self.subTest(cls=cls, form=form):
                    self.assertFalse(has_evidence(f'from {module} import {cls}\nclass {cls}: pass\n' + form))

    def test_span_removal(self):
        self.assertEqual(match_claims('does not model topics'), ([], ['does not model topics']))

    def test_longest_first(self):
        self.assertEqual(match_claims('TOPIC MODELING SYSTEM'), (['topic modeling system'], []))

    def test_substring_collision(self):
        self.assertEqual(match_claims('topical'), (['topic'], []))
        with tempfile.TemporaryDirectory() as d:
            Path(d, 'README.md').write_text('topical')
            self.assertEqual(audit(d)[0][0]['verdict'], 'HIGH')

    def test_fences(self):
        self.assertEqual(match_claims('```text\ntopic modeling\n```'), ([], []))

    def test_inline(self):
        self.assertEqual(match_claims('`topic modeling`'), (['topic modeling'], []))

    def invoke(self, args):
        output, error = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(error):
            try:
                code = main(args)
            except SystemExit as exc:
                code = exc.code
        return code, output.getvalue(), error.getvalue()

    def fixture(self, readme='', source='', test_doc=None):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        (root / 'README.md').write_text(readme)
        (root / 'code.py').write_text(source)
        if test_doc is not None:
            (root / 'tests').mkdir()
            (root / 'tests' / 'fixture.md').write_text(test_doc)
        return root

    def test_cli_good(self):
        root = self.fixture('topic modeling', 'x.extract_topics()')
        self.assertEqual(self.invoke(['audit', str(root)])[:2], (0, 'PASS\n'))

    def test_cli_bad(self):
        root = self.fixture('topic modeling')
        code, out, _ = self.invoke(['audit', str(root)])
        self.assertEqual(code, 2)
        self.assertIn('HIGH documented-topic-modeling-claim-false:', out)

    def test_cli_test_only(self):
        root = self.fixture(test_doc='topic modeling')
        code, out, _ = self.invoke(['audit', str(root)])
        self.assertEqual(code, 0)
        self.assertIn('MEDIUM documented-topic-modeling-claim-false:', out)
        self.assertEqual(self.invoke(['audit', str(root), '--fail-on', 'medium'])[0], 2)

    def test_cli_denial(self):
        root = self.fixture('no topic modeling', 'x.extract_topics()')
        code, out, _ = self.invoke(['audit', str(root)])
        self.assertEqual(code, 2)
        self.assertIn('HIGH documented-no-topic-modeling-claim-false:', out)

    def test_cli_no_claim(self):
        self.assertEqual(self.invoke(['audit', str(self.fixture('ordinary prose'))])[:2], (0, 'PASS\n'))
        self.assertEqual(self.invoke(['audit', str(self.fixture('no topic modeling'))])[0], 0)

    def test_json(self):
        code, out, _ = self.invoke(['audit', str(self.fixture('topics')), '--json'])
        self.assertEqual(code, 2)
        finding = json.loads(out)[0]
        self.assertEqual(finding['verdict'], 'HIGH')
        self.assertTrue(finding['message'])
        self.assertEqual(finding['id'], 'documented-topic-modeling-claim-false')

    def test_version(self):
        self.assertEqual(self.invoke(['--version'])[:2], (0, '0.1.0\n'))

    def test_ignore(self):
        root = self.fixture('topic modeling')
        self.assertEqual(self.invoke(['audit', str(root), '--ignore', 'README'])[0], 0)
        (root / 'code.py').write_text('x.extract_topics()')
        self.assertEqual(self.invoke(['audit', str(root), '--ignore', 'code.py'])[0], 2)

    def test_verbose_and_errors(self):
        root = self.fixture()
        code, _, err = self.invoke(['audit', str(root), '--verbose'])
        self.assertEqual(code, 0)
        self.assertIn('Scope:', err)
        self.assertIn('code.py: no evidence', err)
        self.assertEqual(self.invoke(['audit', str(root / 'missing')])[0], 1)
        self.assertEqual(self.invoke(['audit', '--invalid'])[0], 1)

    def test_split_modules(self):
        root = self.fixture('topic modeling', 'from sklearn.decomposition import NMF\nm=NMF()')
        (root / 'other.py').write_text('from code import m\nm.fit(data)')
        self.assertEqual(audit(root)[0][0]['verdict'], 'HIGH')

    def test_scope_exclusions(self):
        root = self.fixture('topic modeling')
        for directory in ['.git', '__pycache__', '.venv', 'custom_env']:
            target = root / directory
            target.mkdir()
            if directory == 'custom_env':
                (target / 'pyvenv.cfg').write_text('home = /usr/bin')
            (target / 'code.py').write_text('x.extract_topics()')
        self.assertEqual(audit(root)[0][0]['verdict'], 'HIGH')
        (root / 'docs').mkdir()
        (root / 'README.md').write_text('')
        (root / 'docs' / 'nested').mkdir()
        (root / 'docs' / 'nested' / 'guide.md').write_text('topic modeling')
        self.assertEqual(audit(root)[0][0]['verdict'], 'HIGH')

    def test_self_audit(self):
        root = Path(__file__).resolve().parents[1]
        self.assertEqual(match_claims((root / 'README.md').read_text()), ([], []))
        self.assertEqual(audit(root)[0], [])

if __name__ == '__main__':
    unittest.main()
