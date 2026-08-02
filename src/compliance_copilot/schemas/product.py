from pydantic import BaseModel, ConfigDict, Field


class ProductAttributes(BaseModel):
    """Structured extraction from a supplier spec. Every field is optional because
    a vague or empty spec must still produce a valid (mostly-empty) instance —
    that emptiness is the signal the agent uses to recognize insufficient
    information, rather than a special-cased early exit."""

    model_config = ConfigDict(frozen=True)

    product_name: str | None = None
    voltage: str | None = Field(default=None, description='e.g. "220V AC" or "5V DC"')
    frequency: str | None = Field(default=None, description='e.g. "50/60Hz"')
    materials: list[str] = Field(default_factory=list)
    wireless_bands: list[str] = Field(
        default_factory=list, description='e.g. "Bluetooth", "Wi-Fi 2.4GHz"'
    )
    has_batteries: bool | None = None
    intended_user: str | None = Field(
        default=None, description='e.g. "children", "general consumer", "industrial"'
    )
    notes: str | None = None

    @property
    def is_empty(self) -> bool:
        """True when extraction found no *substantive* attribute — the trigger
        for the agent reporting insufficient evidence instead of guessing.

        Deliberately excludes product_name: a bare label ("Widget X200") with no
        voltage, materials, or other technical fact still can't support any
        compliance determination, so a name alone must not count as "not empty."
        """
        return (
            self.voltage is None
            and self.frequency is None
            and not self.materials
            and not self.wireless_bands
            and self.has_batteries is None
            and self.intended_user is None
        )
