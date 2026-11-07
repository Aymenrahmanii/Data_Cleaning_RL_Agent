"""Twenty clean German business table types, generated from a seed.

Each table type is a list of typed columns. ``generate`` returns the clean table with
canonical dtypes: Int64 for integers, float64, datetime64[ns] (dates only) and object for
text and categories, with NaN/NA/NaT for missing values. The first column is always a
unique id.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from dcrl.pools import pool
from dcrl.schema import ColumnSpec, Quantity, TargetSchema

Gen = Callable[[np.random.Generator, int], pd.Series]

NULL_FRACTION = 0.08


@dataclass(frozen=True)
class Column:
    spec: ColumnSpec
    gen: Gen


@dataclass(frozen=True)
class TableType:
    name: str
    description: str
    columns: tuple[Column, ...]

    def __post_init__(self) -> None:
        if not self.columns[0].spec.is_id:
            raise ValueError(f"{self.name}: the first column must be the id")

    @property
    def schema(self) -> TargetSchema:
        return TargetSchema(tuple(c.spec for c in self.columns))

    @property
    def max_rows(self) -> int | None:
        """Generated tables have no row limit."""
        return None

    def generate(self, n_rows: int, rng: np.random.Generator) -> pd.DataFrame:
        data: dict[str, pd.Series] = {}
        for col in self.columns:
            values = col.gen(rng, n_rows).reset_index(drop=True)
            if col.spec.nullable:
                mask = rng.random(n_rows) < NULL_FRACTION
                if n_rows >= 20 and not mask.any():
                    mask[int(rng.integers(n_rows))] = True
                values = values.mask(mask)
            data[col.spec.name] = values
        return pd.DataFrame(data)


# --- column builders -------------------------------------------------------------------


def ident(name: str) -> Column:
    def gen(rng: np.random.Generator, n: int) -> pd.Series:
        start = int(rng.integers(10_000, 500_000))
        return pd.Series(start + np.cumsum(rng.integers(1, 4, size=n)), dtype="Int64")

    return Column(ColumnSpec(name, "int", is_id=True, min_value=10_000.0), gen)


def integer(
    name: str, lo: int, hi: int, *, unit: str | None = None, nullable: bool = False
) -> Column:
    def gen(rng: np.random.Generator, n: int) -> pd.Series:
        return pd.Series(rng.integers(lo, hi + 1, size=n), dtype="Int64")

    spec = ColumnSpec(
        name, "int", unit=unit, nullable=nullable, min_value=float(lo), max_value=float(hi)
    )
    return Column(spec, gen)


def floating(
    name: str,
    lo: float,
    hi: float,
    decimals: int,
    *,
    unit: str | None = None,
    quantity: Quantity | None = None,
    nullable: bool = False,
    log: bool = False,
) -> Column:
    if decimals < 1:
        raise ValueError("float columns keep at least one decimal")

    def gen(rng: np.random.Generator, n: int) -> pd.Series:
        raw = np.exp(rng.uniform(np.log(lo), np.log(hi), n)) if log else rng.uniform(lo, hi, n)
        return pd.Series(np.round(raw, decimals) + 0.0, dtype="float64")

    spec = ColumnSpec(
        name,
        "float",
        unit=unit,
        nullable=nullable,
        quantity=quantity,
        decimals=decimals,
        min_value=float(lo),
        max_value=float(hi),
    )
    return Column(spec, gen)


def money(name: str, lo: float, hi: float, *, nullable: bool = False) -> Column:
    return floating(name, lo, hi, 2, unit="EUR", quantity="money", nullable=nullable, log=True)


def weight(name: str, lo: float, hi: float, *, nullable: bool = False) -> Column:
    return floating(name, lo, hi, 3, unit="kg", quantity="weight", nullable=nullable, log=True)


def length(name: str, lo: float, hi: float, *, nullable: bool = False) -> Column:
    return floating(name, lo, hi, 3, unit="m", quantity="length", nullable=nullable)


def volume(name: str, lo: float, hi: float, *, nullable: bool = False) -> Column:
    return floating(name, lo, hi, 2, unit="l", quantity="volume", nullable=nullable, log=True)


def percent(
    name: str,
    *,
    choices: Sequence[float] | None = None,
    lo: float = 0.0,
    hi: float = 100.0,
    decimals: int = 1,
    nullable: bool = False,
) -> Column:
    def gen(rng: np.random.Generator, n: int) -> pd.Series:
        if choices is not None:
            values = np.asarray(choices, dtype="float64")[rng.integers(0, len(choices), n)]
        else:
            values = np.round(rng.uniform(lo, hi, n), decimals)
        return pd.Series(values + 0.0, dtype="float64")

    spec = ColumnSpec(
        name,
        "float",
        unit="%",
        nullable=nullable,
        quantity="percent",
        decimals=decimals,
        min_value=float(min(choices)) if choices is not None else lo,
        max_value=float(max(choices)) if choices is not None else hi,
    )
    return Column(spec, gen)


def date(
    name: str, start: str, end: str, *, nullable: bool = False, monthly: bool = False
) -> Column:
    first, last = pd.Timestamp(start), pd.Timestamp(end)
    if first.year < 2000 or last.year > 2068:
        raise ValueError("dates must stay in 2000-2068 so two-digit years are unambiguous")
    span = (last - first).days

    def gen(rng: np.random.Generator, n: int) -> pd.Series:
        days = rng.integers(0, span + 1, size=n)
        values = pd.Series(first + pd.to_timedelta(days, unit="D"))
        if monthly:
            values = values.dt.to_period("M").dt.to_timestamp()
        return values.astype("datetime64[ns]")

    return Column(ColumnSpec(name, "date", nullable=nullable), gen)


def category(name: str, values: Sequence[str], *, nullable: bool = False) -> Column:
    options = np.array(values, dtype=object)

    def gen(rng: np.random.Generator, n: int) -> pd.Series:
        return pd.Series(options[rng.integers(0, len(options), n)], dtype=object)

    return Column(ColumnSpec(name, "category", allowed=tuple(values), nullable=nullable), gen)


def text(name: str, pool_name: str, *, nullable: bool = False) -> Column:
    def gen(rng: np.random.Generator, n: int) -> pd.Series:
        options = np.array(pool(pool_name), dtype=object)
        return pd.Series(options[rng.integers(0, len(options), n)], dtype=object)

    return Column(ColumnSpec(name, "str", nullable=nullable), gen)


# --- category values -------------------------------------------------------------------

CITIES = (
    "München", "Köln", "Düsseldorf", "Nürnberg", "Würzburg", "Lübeck", "Berlin", "Hamburg",
    "Stuttgart", "Göttingen", "Saarbrücken", "Osnabrück", "Frankfurt am Main", "Mönchengladbach",
)  # fmt: skip
WAREHOUSES = ("Lager Nord", "Lager Süd", "Zentrallager", "Außenlager Fürth", "Hochregallager")
DEPARTMENTS = (
    "Vertrieb", "Einkauf", "Produktion", "Logistik", "Buchhaltung", "Personal", "IT",
    "Qualitätssicherung", "Forschung & Entwicklung",
)  # fmt: skip
ORDER_STATUS = ("offen", "bestätigt", "versendet", "geliefert", "storniert")
MACHINES = (
    "Fräse 01", "Fräse 02", "Drehbank 03", "Presse 04", "Schweißroboter 05", "Lackieranlage 06",
    "Stanze 07",
)  # fmt: skip
SENSOR_STATUS = ("ok", "Warnung", "Störung")
CARRIERS = ("DHL", "DPD", "GLS", "Hermes", "UPS", "Dachser", "Schenker")
COUNTRIES = (
    "Deutschland", "Österreich", "Schweiz", "Polen", "Tschechien", "Niederlande", "Frankreich",
    "Italien", "Dänemark",
)  # fmt: skip
PROJECTS = (
    "Projekt Alpha", "Projekt Nordlicht", "Projekt Brücke", "Projekt Kompass", "Intern",
    "Kundenservice",
)  # fmt: skip
ACTIVITIES = ("Entwicklung", "Besprechung", "Dokumentation", "Test", "Reise", "Schulung")
MAINTENANCE_TYPES = ("Inspektion", "Wartung", "Reparatur", "Austausch", "Kalibrierung")
RETURN_REASONS = (
    "defekt", "falscher Artikel", "zu spät geliefert", "gefällt nicht", "Transportschaden",
)  # fmt: skip
PRODUCT_CATEGORIES = (
    "Befestigungstechnik", "Elektro", "Sanitär", "Werkzeug", "Hydraulik", "Büromaterial",
    "Arbeitsschutz",
)  # fmt: skip
CAR_MODELS = (
    "VW Passat", "VW Golf", "Audi A4", "BMW 3er", "Mercedes C-Klasse", "Opel Astra",
    "Ford Transit", "VW Crafter",
)  # fmt: skip
EXPENSE_TYPES = ("Reisekosten", "Bewirtung", "Material", "Software", "Schulung", "Porto")
COST_CENTERS = ("KST 1000", "KST 2000", "KST 3100", "KST 4200", "KST 5000")
INSPECTION_RESULTS = ("bestanden", "Nachprüfung", "abgelehnt")
SITES = ("Werk Augsburg", "Werk Bremen", "Werk Dresden", "Werk Lüneburg", "Verwaltung Köln")
VAT = (7.0, 19.0)


# --- the twenty table types ------------------------------------------------------------

_TABLES = [
    TableType("orders", "Customer orders", (
        ident("bestell_nr"),
        date("bestelldatum", "2019-01-01", "2026-10-31"),
        text("kunde", "company"),
        category("lieferort", CITIES),
        integer("menge", 1, 500),
        money("einzelpreis", 0.5, 2500),
        percent("rabatt", choices=(0, 2, 5, 10, 15), nullable=True),
        category("status", ORDER_STATUS),
    )),
    TableType("invoice_lines", "Invoice line items", (
        ident("rechnungs_nr"),
        date("rechnungsdatum", "2020-01-01", "2026-10-31"),
        integer("position", 1, 30),
        text("artikel", "product"),
        integer("menge", 1, 250),
        money("preis_netto", 0.5, 5000),
        percent("mwst_satz", choices=VAT),
    )),
    TableType("inventory", "Stock levels per article", (
        ident("artikel_nr"),
        text("artikel", "product"),
        category("lager", WAREHOUSES),
        integer("bestand", 0, 25_000, nullable=True),
        weight("gewicht", 0.05, 80),
        money("einkaufspreis", 0.2, 3000),
        date("letzte_inventur", "2022-01-01", "2026-10-31"),
    )),
    TableType("staff", "Staff list", (
        ident("personal_nr"),
        text("name", "person"),
        category("abteilung", DEPARTMENTS),
        category("standort", CITIES),
        date("eintrittsdatum", "2000-01-01", "2026-09-30"),
        floating("wochenstunden", 8, 40, 1, unit="h"),
        money("bruttogehalt", 2100, 9800),
    )),
    TableType("sensor_logs", "Machine sensor readings", (
        ident("messung_id"),
        category("maschine", MACHINES),
        date("datum", "2024-01-01", "2026-10-31"),
        floating("temperatur", -10, 120, 1, unit="°C"),
        floating("druck", 0.5, 12, 2, unit="bar"),
        floating("vibration", 0.1, 25, 2, unit="mm/s", nullable=True),
        category("status", SENSOR_STATUS),
    )),
    TableType("deliveries", "Incoming deliveries", (
        ident("lieferschein_nr"),
        date("lieferdatum", "2021-01-01", "2026-10-31"),
        text("lieferant", "company"),
        category("zielort", CITIES),
        weight("gewicht", 1, 2400),
        integer("paletten", 1, 33),
        category("spediteur", CARRIERS),
    )),
    TableType("customers", "Customer master data", (
        ident("kunden_nr"),
        text("firma", "company"),
        text("ansprechpartner", "person", nullable=True),
        text("strasse", "street"),
        category("stadt", CITIES),
        money("umsatz_vorjahr", 500, 2_500_000, nullable=True),
        date("kunde_seit", "2000-01-01", "2026-10-31"),
    )),
    TableType("suppliers", "Supplier master data", (
        ident("lieferanten_nr"),
        text("firma", "company"),
        category("land", COUNTRIES),
        floating("bewertung", 1, 5, 1),
        integer("zahlungsziel", 7, 90, unit="Tage"),
        date("rahmenvertrag_bis", "2025-01-01", "2030-12-31", nullable=True),
        money("jahresvolumen", 1000, 5_000_000),
    )),
    TableType("purchase_orders", "Purchase orders", (
        ident("bestellung_nr"),
        text("lieferant", "company"),
        text("artikel", "product"),
        integer("menge", 1, 10_000),
        money("stueckpreis", 0.05, 1500),
        date("liefertermin", "2022-01-01", "2027-06-30"),
        category("abteilung", DEPARTMENTS),
    )),
    TableType("payroll", "Monthly payroll", (
        ident("abrechnung_id"),
        integer("personal_nr", 1000, 9999),
        date("monat", "2021-01-01", "2026-10-31", monthly=True),
        money("brutto", 1800, 9800),
        money("lohnsteuer", 100, 3000),
        money("sozialabgaben", 300, 2000),
        floating("ueberstunden", 0.5, 40, 1, unit="h", nullable=True),
    )),
    TableType("timesheets", "Time sheet entries", (
        ident("eintrag_id"),
        text("mitarbeiter", "person"),
        date("datum", "2024-01-01", "2026-10-31"),
        category("projekt", PROJECTS),
        floating("stunden", 0.25, 12, 2, unit="h"),
        category("taetigkeit", ACTIVITIES),
    )),
    TableType("maintenance", "Machine maintenance orders", (
        ident("auftrag_nr"),
        category("maschine", MACHINES),
        date("datum", "2020-01-01", "2026-10-31"),
        text("techniker", "person"),
        category("art", MAINTENANCE_TYPES),
        floating("dauer", 0.25, 16, 2, unit="h"),
        money("kosten", 20, 25_000, nullable=True),
    )),
    TableType("shipments", "Outgoing parcels", (
        ident("sendungs_nr"),
        date("versanddatum", "2023-01-01", "2026-10-31"),
        text("empfaenger", "person"),
        category("zielort", CITIES),
        weight("paketgewicht", 0.05, 31.5),
        length("laenge", 0.05, 2.5),
        money("versandkosten", 3.5, 250),
    )),
    TableType("returns", "Customer returns", (
        ident("retoure_nr"),
        integer("bestell_nr", 100_000, 999_999),
        text("artikel", "product"),
        date("retourdatum", "2022-01-01", "2026-10-31"),
        category("grund", RETURN_REASONS),
        money("erstattung", 1, 3000, nullable=True),
    )),
    TableType("product_catalog", "Product catalogue", (
        ident("artikel_nr"),
        text("bezeichnung", "product"),
        category("kategorie", PRODUCT_CATEGORIES),
        money("listenpreis", 0.1, 4000),
        weight("gewicht", 0.01, 60),
        length("laenge", 0.01, 6),
        percent("mwst_satz", choices=VAT),
    )),
    TableType("energy_readings", "Energy meter readings", (
        ident("zaehler_id"),
        category("standort", SITES),
        date("ablesedatum", "2021-01-01", "2026-10-31"),
        floating("verbrauch", 50, 250_000, 1, unit="kWh", log=True),
        money("kosten", 10, 60_000),
        percent("anteil_erneuerbar", lo=0, hi=100),
    )),
    TableType("fleet", "Company vehicles", (
        ident("fahrzeug_id"),
        text("kennzeichen", "plate"),
        category("modell", CAR_MODELS),
        date("erstzulassung", "2012-01-01", "2026-10-31"),
        integer("kilometerstand", 500, 320_000, unit="km"),
        volume("tankvolumen", 40, 95),
        category("standort", SITES),
    )),
    TableType("expenses", "Employee expense claims", (
        ident("beleg_nr"),
        date("belegdatum", "2023-01-01", "2026-10-31"),
        text("mitarbeiter", "person"),
        category("kategorie", EXPENSE_TYPES),
        money("betrag", 1, 4500),
        percent("mwst_satz", choices=(0, 7, 19)),
        category("kostenstelle", COST_CENTERS),
    )),
    TableType("quality_inspections", "Quality inspection results", (
        ident("pruefung_id"),
        date("datum", "2023-01-01", "2026-10-31"),
        text("produkt", "product"),
        text("pruefer", "person"),
        floating("durchmesser", 2, 250, 3, unit="mm"),
        percent("ausschussquote", lo=0, hi=12, decimals=2),
        category("ergebnis", INSPECTION_RESULTS),
    )),
    TableType("warehouse_moves", "Stock movements between warehouses", (
        ident("bewegung_id"),
        date("datum", "2024-01-01", "2026-10-31"),
        text("artikel", "product"),
        category("von_lager", WAREHOUSES),
        category("nach_lager", WAREHOUSES),
        integer("menge", 1, 5000),
        volume("volumen", 0.5, 2000, nullable=True),
    )),
]  # fmt: skip

TABLE_TYPES: dict[str, TableType] = {t.name: t for t in _TABLES}
