from unittest.mock import MagicMock, patch
import pytest
from gpu_energy_cuda.energy_monitor import EnergyMonitor

# Mocking EnergyMonitor
@patch("gpu_energy_cuda.energy_monitor.EnergyMonitor", autospec=True)
def test_energy_monitor(mock_energy_monitor):
    # Simula que la clase devuelve valores específicos sin necesitar GPU
    mock_instance = mock_energy_monitor.return_value
    mock_instance.get_energy.return_value = 42.0  # Un valor de prueba

    energy_monitor = EnergyMonitor()
    assert energy_monitor.get_energy() == 42.0

