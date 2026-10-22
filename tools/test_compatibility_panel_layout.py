"""Adapter/source wiring checks; actual UI interaction lives in shared_panel_tests."""
from pathlib import Path
import unittest
ROOT = Path(__file__).resolve().parents[1]

class CompatibilityPanelLayoutTests(unittest.TestCase):
    def test_one_shared_dashboard_owns_compatibility_layout(self):
        source = (ROOT/'src/integration/compatibility/controller.h').read_text()
        panel = source.split('inline void panel(', 1)[1].split('inline void destroy(', 1)[0]
        self.assertEqual(panel.count('phoenix_ui::draw_controls(ui)'), 1)
        self.assertIn('ui.setup_details =', panel)
        self.assertIn('draw_transport(runtime, value)', panel)
        self.assertLess(panel.index('phoenix_ui::draw_controls(ui)'), panel.index('apply_ui_edits(runtime, value, ui)'))
        self.assertNotIn('phoenix_ui::draw_timing', panel)

    def test_saved_execution_identifiers_and_eligibility_remain_shared(self):
        source = (ROOT/'src/integration/compatibility/backbuffer_route.h').read_text()
        for value in ('in_process','helper','portable','wow64_library','wow64_driver','auto'):
            self.assertIn('"'+value+'"', source)
        for number in range(1, 5): self.assertNotIn(f'#{number}', source)
        controller = (ROOT/'src/integration/compatibility/controller.h').read_text()
        transport = controller.split('inline void draw_transport(', 1)[1].split('inline void panel(', 1)[0]
        self.assertIn('available_execution_preferences(value.selectable_capabilities)', transport)
        self.assertIn('execution_preference_label(choice)', transport)

if __name__ == '__main__': unittest.main()
