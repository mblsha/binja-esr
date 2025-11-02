from sc62015.pysc62015.stepper import (
    CPUStepper,
    CPURegistersSnapshot,
)


def test_stepper_executes_nop_and_advances_pc() -> None:
    stepper = CPUStepper()
    initial = CPURegistersSnapshot(pc=0x100, ba=0, i=0, x=0, y=0, u=0, s=0, f=0)
    memory = {0x100: 0x00}

    result = stepper.step(initial, memory)

    assert result.instruction_name == "NOP"
    assert result.instruction_length == 1
    assert result.registers.pc == 0x101
    assert result.changed_registers == {"PC": (0x100, 0x101)}
    assert not result.memory_writes
    assert result.memory_image[0x100] == 0x00


def test_stepper_wait_clears_i_register() -> None:
    stepper = CPUStepper()
    initial = CPURegistersSnapshot(pc=0x200, i=0x1234)
    memory = {0x200: 0xEF}

    result = stepper.step(initial, memory)

    assert result.instruction_name == "WAIT"
    assert result.registers.i == 0
    assert result.registers.pc == 0x201
    assert result.changed_registers["I"] == (0x1234, 0)


def test_stepper_sc_sets_carry_flag() -> None:
    stepper = CPUStepper()
    initial = CPURegistersSnapshot(pc=0x300, f=0x00)
    memory = {0x300: 0x97}

    result = stepper.step(initial, memory)

    assert result.instruction_name == "SC"
    assert result.registers.f & 0x01 == 0x01
    assert result.registers.pc == 0x301
    assert result.changed_registers["F"] == (0x00, result.registers.f)
