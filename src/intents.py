"""Intent taxonomy for the Delta support agent, derived from clustering ~6k
customer opening tweets (see notebooks / DECISIONS.md). Kept deliberately small.
"""
from __future__ import annotations

INTENTS: dict[str, str] = {
    "flight_disruption": "Delay, cancellation, diversion, missed connection, IRROPS, "
    "being stranded/rebooked due to weather or mechanical issues.",
    "baggage": "Lost, delayed, or damaged baggage; baggage fees, allowance, or policy.",
    "booking_reservation": "Anything about a booking or fare: new reservations, "
    "schedules/routes, prices, changes, cancellations, refunds, name corrections.",
    "seat_upgrade": "Seat assignment or change, upgrade lists, cabin (Comfort+, First, "
    "Premium Select), standby.",
    "checkin_boarding": "Check-in problems, boarding passes, gate/boarding issues, "
    "kiosk / app / website errors during travel.",
    "loyalty_miles": "SkyMiles, Medallion status, miles/points not posting, Sky Club, "
    "companion certificates, co-brand credit-card benefits.",
    "complaint": "General dissatisfaction with service, staff behaviour, or an unresolved "
    "prior issue — no specific actionable request.",
    "compliment": "Praise for staff, crew, or the airline; positive feedback.",
    "other": "Off-topic, spam, press/news, or not addressed to support.",
}

LABELS = list(INTENTS)

# Intent risk class for the router (Phase 5): 'high' intents almost always need a
# human because they require account/PNR-level action; 'low' are often auto-answerable.
RISK_CLASS: dict[str, str] = {
    "flight_disruption": "high",
    "baggage": "high",
    "booking_reservation": "high",
    "seat_upgrade": "high",
    "checkin_boarding": "medium",
    "loyalty_miles": "high",
    "complaint": "medium",
    "compliment": "low",
    "other": "low",
}
