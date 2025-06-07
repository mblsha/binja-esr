class Architecture:
    name: str

    def __getitem__(self, name: str) -> "Architecture":
        ...

RegisterName = str
IntrinsicName = str
FlagName = str
