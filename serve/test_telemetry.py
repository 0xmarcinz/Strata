"""serve/test_telemetry.py - the CPU, drive and fan readings the kernel exposes through hwmon, from fake trees (no
GPU, no model, no psutil sensor source).

    python -m unittest serve.test_telemetry -v
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve import telemetry  # noqa: E402


def chip(root, folder, name, temps=(), fans=()):
    """One hwmon folder holding the files the kernel writes: temps [(label, °C)], fans [(label, rpm)]."""
    d = Path(root, folder)
    d.mkdir(parents=True, exist_ok=True)
    (d / "name").write_text(name)
    for i, (label, celsius) in enumerate(temps, 1):
        (d / f"temp{i}_input").write_text(str(int(round(celsius * 1000))))
        if label:
            (d / f"temp{i}_label").write_text(label)
    for i, (label, rpm) in enumerate(fans, 1):
        (d / f"fan{i}_input").write_text(str(rpm))
        if label:
            (d / f"fan{i}_label").write_text(label)
    return d


class Sensors(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name, "hwmon")
        self.root.mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def test_amd_cpu_temperature_is_tctl(self):
        chip(self.root, "hwmon0", "acpitz", temps=[("", 16.8)])          # a case probe on a desktop board
        chip(self.root, "hwmon5", "k10temp", temps=[("Tctl", 45.375), ("Tccd1", 45.0)])
        self.assertEqual(telemetry.cpu_temperature(str(self.root)), 45.4)

    def test_intel_cpu_temperature_is_the_package(self):
        chip(self.root, "hwmon3", "coretemp", temps=[("Core 0", 38.0), ("Package id 0", 41.0)])
        self.assertEqual(telemetry.cpu_temperature(str(self.root)), 41.0)

    def test_cpu_chip_with_other_labels_still_gives_its_first_sensor(self):
        chip(self.root, "hwmon1", "k10temp", temps=[("Sensor 9", 50.0)])
        self.assertEqual(telemetry.cpu_temperature(str(self.root)), 50.0)

    def test_no_cpu_sensor_is_a_dash_not_a_guess(self):
        chip(self.root, "hwmon0", "acpitz", temps=[("", 16.8)])          # ACPI alone: not a CPU reading
        self.assertIsNone(telemetry.cpu_temperature(str(self.root)))

    def test_psutil_is_the_fallback_when_the_chip_is_there(self):
        entry = mock.Mock(label="Tctl", current=44.0)
        self.assertEqual(telemetry.cpu_temperature(str(self.root), sensors={"k10temp": [entry]}), 44.0)

    def test_drive_temperatures(self):
        block = Path(self.tmp.name, "block")
        nvme = block / "nvme0n1" / "device"
        (nvme / "hwmon1").mkdir(parents=True)
        (nvme / "model").write_text("INTEL SSDPED1D280GA      \n")
        (nvme / "hwmon1" / "name").write_text("nvme")
        (nvme / "hwmon1" / "temp1_input").write_text("38850")
        (nvme / "hwmon1" / "temp1_label").write_text("Composite")
        (nvme / "hwmon1" / "temp3_input").write_text("49850")            # Sensor 2: hotter, not the drive's number
        sata = block / "sda" / "device"
        (sata / "hwmon" / "hwmon2").mkdir(parents=True)                  # drivetemp
        (sata / "model").write_text("CT1000MX500SSD1")
        (sata / "hwmon" / "hwmon2" / "name").write_text("drivetemp")
        (sata / "hwmon" / "hwmon2" / "temp1_input").write_text("31000")
        (block / "sdb" / "device").mkdir(parents=True)                   # a drive with no sensor node
        (block / "dm-0").mkdir()                                         # not a drive
        (block / "loop0").mkdir()
        got = telemetry.drive_temperatures(str(block))
        self.assertEqual(got, [{"name": "nvme0n1", "model": "INTEL SSDPED1D280GA", "temp": 38.9},
                               {"name": "sda", "model": "CT1000MX500SSD1", "temp": 31.0}])

    def test_board_fans(self):
        chip(self.root, "hwmon2", "nct6775", temps=[("SYSTIN", 30.0), ("CPUTIN", 45.0)],
             fans=[("CPU Fan", 1180), ("Chassis Fan #1", 820)])
        chip(self.root, "hwmon6", "amdgpu", fans=[("", 1500)])           # the card's own fan: its reader's job
        self.assertEqual(telemetry.fan_speeds(str(self.root)),
                         [{"chip": "nct6775", "label": "CPU Fan", "name": None, "rpm": 1180,
                           "duty": None, "role": "cpu"},
                          {"chip": "nct6775", "label": "Chassis Fan #1", "name": None, "rpm": 820,
                           "duty": None, "role": None}])

    def test_the_config_names_a_fan_and_its_pwm_is_a_percentage(self):
        d = chip(self.root, "hwmon7", "it8688_6008090b", fans=[("", 780), ("", 0)])
        (d / "pwm1").write_text("76\n")                                # 30% of 255
        (d / "pwm2").write_text("255\n")
        names = {"it8688/fan1": "Bottom intakes (3)", "it8688/fan2": "Top exhausts (2)"}
        got = telemetry.fan_speeds(str(self.root), names=names)
        self.assertEqual([(f["name"], f["rpm"], f["duty"]) for f in got],
                         [("Bottom intakes (3)", 780, 30), ("Top exhausts (2)", 0, 100)])

    def test_a_fan_named_as_the_cpu_is_marked_for_the_cpu_temperature_card(self):
        chip(self.root, "hwmon7", "it8688_6008090b", fans=[("", 850), ("", 700)])
        got = telemetry.fan_speeds(str(self.root), names={"it8688/fan1": "CPU fan"})
        self.assertEqual([f["role"] for f in got], ["cpu", None])

    def test_a_names_key_that_matches_nothing_leaves_the_label(self):
        chip(self.root, "hwmon7", "it8688_6008090b", fans=[("", 900)])
        got = telemetry.fan_speeds(str(self.root), names={"it8689/fan1": "Nope", "it8688/fan2": "Nope"})
        self.assertEqual((got[0]["name"], got[0]["label"]), (None, "fan1"))

    def test_no_board_driver_reads_nothing(self):
        chip(self.root, "hwmon0", "k10temp", temps=[("Tctl", 45.0)])
        self.assertEqual(telemetry.fan_speeds(str(self.root)), [])


class Sample(unittest.TestCase):
    def test_a_sample_carries_the_cpu_the_drives_the_fans_and_the_cards_own_fan(self):
        class FakeGpu:
            def ok(self): return True
            def name(self): return "Fake GPU"
            def read(self): return {"util": 5.0, "temp": 40.0, "fan": [0, 30], "mem_used": 1, "mem_total": 2}

        with mock.patch.object(telemetry, "gpu_reader", lambda index=0, amd=False: FakeGpu()), \
             mock.patch.object(telemetry, "cpu_temperature", lambda *a, **k: 45.4), \
             mock.patch.object(telemetry, "drive_temperatures",
                               lambda *a, **k: [{"name": "nvme0n1", "model": "X", "temp": 38.9}]), \
             mock.patch.object(telemetry, "fan_speeds",
                               lambda *a, **k: [{"chip": "nct6775", "label": "CPU Fan", "rpm": 1180}]):
            t = telemetry.Telemetry(extra=lambda: {"tok_s": 1.0})
            try:
                s = t.sample()
            finally:
                t.close()
        self.assertEqual(s["cpu_temp"], 45.4)
        self.assertEqual(s["drives"], [{"name": "nvme0n1", "model": "X", "temp": 38.9}])
        self.assertEqual(s["fans"], [{"chip": "nct6775", "label": "CPU Fan", "rpm": 1180}])
        self.assertEqual(s["gpu_temp"], 40.0)
        self.assertEqual(s["gpu_fan"], 30)               # the fan list, as its fastest fan


if __name__ == "__main__":
    unittest.main()
