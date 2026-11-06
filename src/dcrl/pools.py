"""Fixed value pools (names, companies, streets, products, plates).

Faker is slow per call, so each pool is built once from a fixed seed and the tables sample
from it with numpy. Pool contents depend on the Faker version, which uv.lock pins.
"""

from __future__ import annotations

import random
from functools import cache

from faker import Faker

POOL_SIZE = 1500
MERGE_SEPARATORS = (" / ", " | ")

_FAKER_POOLS = {
    "person": ("name", 11),
    "company": ("company", 12),
    "street": ("street_address", 13),
}

_PRODUCT_NOUNS = (
    "Schraube", "Mutter", "Unterlegscheibe", "Dübel", "Kabelbinder", "Schlauch", "Ventil",
    "Dichtung", "Kugellager", "Zahnrad", "Keilriemen", "Filter", "Sensor", "Schalter", "Relais",
    "Steckdose", "Leuchte", "Akku", "Netzteil", "Motor", "Pumpe", "Halterung", "Winkel",
    "Profil", "Platte", "Rohr", "Flansch", "Kupplung", "Druckfeder", "Bolzen",
)  # fmt: skip
_PRODUCT_SPECS = (
    "M6", "M8", "M10", "Edelstahl", "verzinkt", "schwarz", "weiß", "groß", "klein", "Typ A",
    "Typ B", "Typ C", "DN 20", "DN 50", "12 V", "24 V", "230 V", "Größe 2", "Größe 3", "Premium",
)  # fmt: skip
_PLATE_PREFIXES = ("M", "B", "HH", "K", "S", "F", "N", "DD", "L", "H", "A", "FÜ", "LÜ", "GÖ")


def mojibake(text: str) -> str:
    """UTF-8 bytes read as Windows-1252: München -> MÃ¼nchen."""
    return text.encode("utf-8").decode("cp1252")


def mojibake_safe(text: str) -> bool:
    try:
        return mojibake(text).encode("cp1252").decode("utf-8") == text
    except UnicodeError:
        return False


def usable(text: str) -> bool:
    """Values every corruption can break and undo exactly."""
    return (
        bool(text)
        and text == " ".join(text.split())
        and mojibake_safe(text)
        and not any(sep in text for sep in MERGE_SEPARATORS)
    )


def _faker_pool(method: str, seed: int) -> tuple[str, ...]:
    fake = Faker("de_DE")
    fake.seed_instance(seed)
    make = getattr(fake, method)
    values: set[str] = set()
    for _ in range(POOL_SIZE * 10):
        value = " ".join(str(make()).split())
        if usable(value):
            values.add(value)
        if len(values) >= POOL_SIZE:
            break
    return tuple(sorted(values))


def _plates(seed: int) -> tuple[str, ...]:
    r = random.Random(seed)
    letters = "ABCDEFGHJKLMNPRSTUVWXYZ"
    values = {
        f"{r.choice(_PLATE_PREFIXES)}-{r.choice(letters)}{r.choice(letters)} {r.randint(1, 9999)}"
        for _ in range(POOL_SIZE * 2)
    }
    return tuple(sorted(values))[:POOL_SIZE]


@cache
def pool(name: str) -> tuple[str, ...]:
    if name in _FAKER_POOLS:
        method, seed = _FAKER_POOLS[name]
        return _faker_pool(method, seed)
    if name == "product":
        return tuple(f"{n} {s}" for n in _PRODUCT_NOUNS for s in _PRODUCT_SPECS)
    if name == "plate":
        return _plates(14)
    raise KeyError(f"unknown pool {name!r}")


POOL_NAMES = (*_FAKER_POOLS, "product", "plate")
