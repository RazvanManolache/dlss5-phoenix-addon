"""Guard the short front page and moved-document link behavior."""
from pathlib import Path
import unittest
from readme_summary import benchmark_block, relative_links, replace_block

ROOT = Path(__file__).resolve().parents[1]


class ReadmeSummary(unittest.TestCase):
    def test_front_page_remains_an_overview(self):
        text = (ROOT / 'README.md').read_text(encoding='utf-8')
        self.assertLessEqual(len(text.splitlines()), 100)
        self.assertNotIn('<details>', text)
        self.assertNotIn('```', text)
        self.assertLess(text.index('## Native RTX:'), text.index('## Features'))
        for guide in ['INSTALLATION', 'USER_GUIDE', 'FEATURES', 'COMPATIBILITY', 'PERFORMANCE', 'DEVELOPMENT', 'PROJECT']:
            self.assertIn('docs/' + guide + '.md', text)
            self.assertTrue((ROOT / 'docs' / (guide + '.md')).is_file())

    def test_generated_summary_does_not_reintroduce_detailed_sections(self):
        get = lambda route, model: {
            'gpu': {'median': 3.5 if model == 'original' else 2.25},
            'memory': {'device_local_bytes': 2**20 if route == 'phoenix' else 2 * 2**20,
                       'host_working_set_bytes': 3 * 2**20 if route == 'phoenix' else 4 * 2**20,
                       'host_private_bytes': 999 * 2**20}}
        block = benchmark_block(get, 'test date', 'median')
        self.assertLess(len(block.splitlines()), 24)
        self.assertNotIn('![', block)
        self.assertNotIn('Build from source', block)
        self.assertIn('| Original | 3.500 | 3.500 | 1.0 | 2.0 |', block)
        self.assertIn('3.0 / 4.0 MiB for Original', block)
        self.assertNotIn('999', block)  # Private commitment must not be called resident RAM.
        authored = 'Intro\n' + block + '\nCustom feature list\n'
        self.assertEqual(replace_block(authored, block), authored)
        with self.assertRaises(RuntimeError): replace_block('missing markers', block)

    def test_moving_prose_rebases_links_and_html_but_preserves_code(self):
        text = ('[Controls](docs/NEURAL_CONTROLS.md#controls)\n'
                '<img src="docs/features/overlay-play.png">\n'
                '[License](LICENSE) [Web](https://example.com/a) [Local](#heading)\n'
                '```markdown\n[Root command example](docs/README.md)\n```\n')
        moved = relative_links(text, 'README.md', 'docs/USER_GUIDE.md')
        self.assertIn('(NEURAL_CONTROLS.md#controls)', moved)
        self.assertIn('src="features/overlay-play.png"', moved)
        self.assertIn('(../LICENSE)', moved)
        self.assertIn('(https://example.com/a)', moved)
        self.assertIn('(#heading)', moved)
        self.assertIn('[Root command example](docs/README.md)', moved)

    def test_later_regression_cannot_keep_a_stale_parity_claim(self):
        get = lambda route, model: {'gpu': {'median': 4. if route == 'phoenix' else 2.},
                                   'memory': {'device_local_bytes': 2**20, 'host_working_set_bytes': 2**20}}
        block = benchmark_block(get, 'later', 'median')
        self.assertNotIn('Performance on par', block)
        self.assertIn('4.000', block)


if __name__ == '__main__': unittest.main()
