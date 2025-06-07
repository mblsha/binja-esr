class Architecture:
    name: str

    def __getitem__(self, name: str) -> "Architecture":
        ...


class RegisterName(str):
    ...


class IntrinsicName(str):
    ...


class FlagName(str):
    ...
