"""Focused source contracts for renderer-first layout and stable saved choices."""
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]


class CompatibilityPanelLayoutTests(unittest.TestCase):
    def setUp(self):
        source=(ROOT/'src/integration/compatibility/controller.h').read_text()
        self.panel=source.split('inline void panel(',1)[1].split('inline void destroy(',1)[0]

    def test_renderer_controls_and_edits_precede_transport(self):
        controls=self.panel.index('phoenix_ui::draw_controls(ui)')
        edits=self.panel.index('apply_ui_edits(runtime, value, ui)')
        transport=self.panel.index('ImGui::CollapsingHeader("Advanced transport")')
        self.assertLess(controls,edits)
        self.assertLess(edits,transport)
        self.assertEqual(self.panel.count('phoenix_ui::draw_controls(ui)'),1)

    def test_execution_labels_have_no_investigation_numbers(self):
        labels=self.panel.split('const char* choices[]{',1)[1].split('};',1)[0]
        self.assertNotIn('#',labels)
        for name in ('Automatic','In-process native D3D12','External x64 helper',
                     'In-process portable (experimental)','x64 library inside x86 (experimental)',
                     'x86 Phoenix + native driver (experimental)'):
            self.assertIn('"'+name+'"',labels)

    def test_status_reasons_have_no_investigation_numbers(self):
        route=(ROOT/'src/integration/compatibility/backbuffer_route.h').read_text()
        for number in range(1,5):
            self.assertNotIn(f'#{number}',route)

    def test_saved_execution_identifiers_are_unchanged(self):
        for value in ('in_process','helper','portable','wow64_library','wow64_driver','auto'):
            self.assertIn('"'+value+'"',self.panel)


if __name__=='__main__': unittest.main()
