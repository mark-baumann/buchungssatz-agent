"""
Booking Engine — LLM-basierte Buchungssatz-Generierung aus Rechnungstexten.

Verwendet LangChain + OpenAI für:
- Extraktion von Rechnungsdaten
- Generierung von Buchungssätzen nach SKR 03/04
- Human-in-the-Loop Verbesserung
"""
from __future__ import annotations

import json
import re
from typing import Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field


class BookingResult(BaseModel):
    """Ergebnis einer Buchungssatz-Generierung."""
    buchungssatz: str = Field(description="Der vollständige Buchungssatz (Soll an Haben)")
    erlaeuterung: str = Field(description="Ausführliche Erläuterung des Buchungssatzes")
    konten: list[dict] = Field(default_factory=list, description="Liste der verwendeten Konten mit Beträgen")
    rechnungsbetrag: float = Field(default=0.0, description="Gesamtbetrag der Rechnung")
    buchungstyp: Optional[str] = Field(default=None, description="Typ der Buchung (Eingangsrechnung, Ausgangsrechnung, etc.)")
    confidence: float = Field(default=0.0, description="KI-Konfidenz (0-1)")
    hinweise: Optional[str] = Field(default=None, description="Zusätzliche Hinweise oder Warnungen")


SYSTEM_PROMPT = """Du bist ein erfahrener Buchhalter und Steuerberater mit Spezialisierung auf das Gesundheitswesen (SGB V, SGB XI, Pflege, Krankenhäuser).

Deine Aufgabe: Analysiere den Rechnungstext und erstelle einen korrekten Buchungssatz nach SKR 03 (Standardkontenrahmen für den Groß- und Einzelhandel) oder SKR 04 (für Körperschaften/Pflegeeinrichtungen).

## Wichtige Konten für das Gesundheitswesen (SKR 03/04):

### Aufwandskonten (Soll):
- 4200-4299: Raumkosten / Miete
- 4400-4499: Sonstige betriebliche Aufwendungen
- 4600-4699: Werbe-/Reisekosten
- 4700-4799: Fahrzeugkosten
- 4900-4999: Sonstige Kosten
- 6000-6099: Aufwendungen für Roh-, Hilfs- und Betriebsstoffe
- 6100-6199: Aufwendungen für bezogene Leistungen (Fremdleistungen)
- 6200-6299: Personalaufwand
- 6300-6399: Abschreibungen
- 6800-6899: Sonstige betriebliche Aufwendungen
- 7000-7099: Zinsaufwendungen

### Spezifische Konten Pflege/Gesundheitswesen:
- 4200: Pflegeleistungen / Betreuungsleistungen
- 4400: Entlastungsleistungen nach §45b SGB XI
- 6100: Fremdleistungen Pflege
- 6120: Ärztliche Leistungen / Honorarärzte
- 6130: Therapieleistungen

### Aktivkonten (Soll bei Zugang):
- 1400-1499: Forderungen aus Lieferungen und Leistungen
- 1500-1599: Sonstige Vermögensgegenstände
- 1600-1699: Bank / Kasse

### Passivkonten (Haben bei Zugang):
- 1600-1699: Verbindlichkeiten aus Lieferungen und Leistungen
- 1700-1799: Sonstige Verbindlichkeiten
- 1800-1899: Bankdarlehen

### Ertragskonten (Haben):
- 8000-8099: Umsatzerlöse
- 8100-8199: Sonstige betriebliche Erträge
- 8200-8299: Erlöse aus Pflegeleistungen
- 8300-8399: Erlöse aus Betreuungsleistungen
- 8400-8499: Erstattungen von Krankenkassen

### Umsatzsteuer:
- 1776: Umsatzsteuer (Vorsteuer)
- 3806: Umsatzsteuer (Umsatzsteuer auf Erlöse)

## Regeln:
1. Bei Eingangsrechnungen (du bekommst eine Rechnung): Aufwandskonto an Verbindlichkeiten (Kreditor)
2. Bei Ausgangsrechnungen (du stellst eine Rechnung): Forderungen (Debitor) an Ertragskonto
3. Prüfe, ob Umsatzsteuer ausgewiesen ist — wenn ja, Vorsteuer/Umsatzsteuer buchen
4. Bei Pflegeleistungen nach SGB V/XI: Prüfe, ob es sich um steuerfreie Leistungen handelt (§4 Nr. 16 UStG)
5. Verwende realistische SKR-Kontonummern
6. Gib den Buchungssatz im Format "Sollkonto (Betrag) an Habenkonto (Betrag)" an

## Ausgabeformat:
Antworte NUR mit einem JSON-Objekt (kein Markdown, kein Codeblock):

{
  "buchungssatz": "4400 Entlastungsleistungen (480,21 €) an 1600 Verbindlichkeiten LuL (480,21 €)",
  "erlaeuterung": "Ausführliche Erläuterung...",
  "konten": [
    {"konto": "4400", "bezeichnung": "Entlastungsleistungen §45b SGB XI", "soll": 480.21, "haben": 0, "typ": "Aufwand"},
    {"konto": "1600", "bezeichnung": "Verbindlichkeiten aus Lieferungen und Leistungen", "soll": 0, "haben": 480.21, "typ": "Passiv"}
  ],
  "rechnungsbetrag": 480.21,
  "buchungstyp": "Eingangsrechnung",
  "confidence": 0.95,
  "hinweise": "Keine Umsatzsteuer — Pflegeleistungen nach §4 Nr. 16 UStG steuerfrei"
}
"""


class BookingEngine:
    """LLM-basierte Buchungssatz-Engine."""

    def __init__(
        self,
        model: str = "gpt-4o",
        openai_api_key: Optional[str] = None,
        openai_base_url: Optional[str] = None,
    ):
        kwargs = {"model": model}
        if openai_api_key:
            kwargs["openai_api_key"] = openai_api_key
        if openai_base_url:
            kwargs["openai_api_base"] = openai_base_url

        self.llm = ChatOpenAI(temperature=0.1, **kwargs)

    def _parse_response(self, text: str) -> BookingResult:
        """Parst die LLM-Antwort in ein BookingResult."""
        # Entferne mögliche Markdown-Codeblöcke
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # Fallback: versuche JSON aus dem Text zu extrahieren
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                data = json.loads(match.group())
            else:
                return BookingResult(
                    buchungssatz="FEHLER: Konnte Antwort nicht parsen",
                    erlaeuterung=text[:500],
                    confidence=0.0,
                )

        return BookingResult(
            buchungssatz=data.get("buchungssatz", ""),
            erlaeuterung=data.get("erlaeuterung", ""),
            konten=data.get("konten", []),
            rechnungsbetrag=float(data.get("rechnungsbetrag", 0)),
            buchungstyp=data.get("buchungstyp"),
            confidence=float(data.get("confidence", 0.0)),
            hinweise=data.get("hinweise"),
        )

    def generate_booking(self, invoice_text: str) -> BookingResult:
        """Generiert einen Buchungssatz aus einem Rechnungstext."""
        user_prompt = f"""Analysiere die folgende Rechnung und erstelle den korrekten Buchungssatz:

--- RECHNUNGSTEXT ---
{invoice_text}
--- ENDE RECHNUNGSTEXT ---

Erstelle den Buchungssatz als JSON. Beachte:
- Handelt es sich um eine Eingangs- oder Ausgangsrechnung?
- Welche Konten sind betroffen?
- Ist Umsatzsteuer relevant?
- Bei Pflege-/Gesundheitsleistungen: Prüfe Steuerfreiheit nach §4 Nr. 16 UStG"""

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]

        response = self.llm.invoke(messages)
        return self._parse_response(response.content)

    def improve_with_feedback(
        self,
        invoice_text: str,
        current_result: BookingResult,
        feedback: str,
    ) -> BookingResult:
        """Verbessert einen Buchungssatz basierend auf Benutzer-Feedback."""
        user_prompt = f"""Der vorherige Buchungssatz war:

Buchungssatz: {current_result.buchungssatz}
Erläuterung: {current_result.erlaeuterung}
Konten: {json.dumps(current_result.konten, ensure_ascii=False)}

Der Benutzer hat folgendes Feedback gegeben:
--- FEEDBACK ---
{feedback}
--- ENDE FEEDBACK ---

Original-Rechnungstext:
--- RECHNUNG ---
{invoice_text}
--- ENDE RECHNUNG ---

Bitte erstelle einen VERBESSERTEN Buchungssatz, der das Feedback berücksichtigt.
Antworte NUR mit dem JSON-Objekt."""

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]

        response = self.llm.invoke(messages)
        return self._parse_response(response.content)
