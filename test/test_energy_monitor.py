import unittest
from gpu_energy_cuda.energy_monitor import EnergyMonitor
import pynvml

class TestEnergyMonitor(unittest.TestCase):
    """
    Unit tests for the EnergyMonitor class.
    """

    def setUp(self):
        """
        Initializes the EnergyMonitor instance before each test.
        """
        self.monitor = EnergyMonitor()

    def test_energy_monitor_initialization(self):
        """
        Tests if the EnergyMonitor initializes correctly without errors.
        """
        self.assertIsNotNone(self.monitor.device_handle)

    def test_get_energy(self):
        """
        Tests if the get_energy method returns a valid numerical value.
        """
        energy = self.monitor.get_energy()
        self.assertIsInstance(energy, float)
        self.assertGreaterEqual(energy, 0)  # Energy should never be negative

    def tearDown(self):
        """
        Shuts down NVML after tests.
        """
        self.monitor.close()

if __name__ == "__main__":
    unittest.main()
