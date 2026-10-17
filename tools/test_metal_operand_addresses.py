import unittest
from generate_metal_operators import generate


class MetalOperandAddresses(unittest.TestCase):
    def test_logical_addresses_become_device_pointers(self):
        source = generate()
        self.assertNotIn('OperandAddress', source)
        for name in ('left_input', 'right_input', 'output', 'auxiliary', 'secondary_output'):
            self.assertIn('device void* '+name+' = nullptr', source)
        self.assertNotIn('operand_address.h', source)


if __name__ == '__main__':
    unittest.main()
