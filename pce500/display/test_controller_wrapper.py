"""Test the HD61202Controller wrapper for emulator compatibility."""

from PIL import Image

from . import HD61202Controller


class TestHD61202Controller:
    """Test the controller wrapper."""

    def test_initialization(self):
        """Test basic initialization."""
        controller = HD61202Controller()
        assert len(controller.chips) == 2
        assert controller.width == 240
        assert controller.height == 32
        assert controller.display_on == [False, False]

    def test_write_instruction_both_chips(self):
        """Test writing instruction to both chips."""
        controller = HD61202Controller()

        # Turn on both displays
        controller.write(0x2000, 0x3F)  # ON instruction to both chips
        assert controller.chips[0].state.on is True
        assert controller.chips[1].state.on is True

    def test_write_instruction_single_chip(self):
        """Test writing instruction to single chip."""
        controller = HD61202Controller()

        # Turn on left chip only
        controller.write(0x2008, 0x3F)  # CS=10 (LEFT)
        assert controller.chips[0].state.on is True
        assert controller.chips[1].state.on is False

        # Turn on right chip
        controller.write(0x2004, 0x3F)  # CS=01 (RIGHT)
        assert controller.chips[0].state.on is True
        assert controller.chips[1].state.on is True

    def test_write_data(self):
        """Test writing data."""
        controller = HD61202Controller()

        # Set page and address
        controller.write(0x2000, 0xB8)  # SET_PAGE 0
        controller.write(0x2000, 0x40)  # SET_Y_ADDRESS 0

        # Write data
        controller.write(0x2002, 0xFF)  # Data write
        assert controller.chips[0].vram[0][0] == 0xFF
        assert controller.chips[1].vram[0][0] == 0xFF

    def test_reset(self):
        """Test reset functionality."""
        controller = HD61202Controller()
        controller.chips[0].state.on = True
        controller.chips[0].vram[0][0] = 0xFF

        controller.reset()

        assert controller.chips[0].state.on is False
        assert controller.chips[0].vram[0][0] == 0

    def test_get_display_buffer(self):
        """Test display buffer generation."""
        controller = HD61202Controller()

        # Turn on displays
        controller.chips[0].state.on = True
        controller.chips[1].state.on = True

        # Write test pattern touching each display segment.
        controller.chips[1].vram[0][0] = 0x01  # Right chip, top half -> column 0
        controller.chips[0].vram[0][0] = 0x01  # Left chip, top half -> column 64
        controller.chips[0].vram[4][55] = 0x01  # Left chip, bottom half -> column 120
        controller.chips[1].vram[4][0] = 0x01  # Right chip, bottom half -> column 239

        buffer = controller.get_display_buffer()
        assert buffer.shape == (32, 240)
        # Bit set in VRAM corresponds to a dark pixel (value 0) in the buffer.
        assert buffer[0, 0] == 0
        assert buffer[0, 64] == 0
        assert buffer[0, 120] == 0
        assert buffer[0, 239] == 0

    def test_get_combined_display(self):
        """Test combined display image generation."""
        controller = HD61202Controller()

        # Write test pattern
        controller.chips[0].vram[0][0] = 0xFF
        controller.chips[1].vram[0][0] = 0xFF

        image = controller.get_combined_display(zoom=1)
        assert isinstance(image, Image.Image)
        assert image.width == 240
        assert image.height == 32

    def test_perfetto_enable(self):
        """Test Perfetto enable/disable."""
        controller = HD61202Controller()
        controller.set_perfetto_enabled(True)
        assert controller.perfetto_enabled is True

        controller.set_perfetto_enabled(False)
        assert controller.perfetto_enabled is False
