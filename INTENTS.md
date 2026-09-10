# Intent taxonomy — Delta support agent

Derived from clustering ~6,000 customer opening tweets to @Delta (TF-IDF + KMeans,
k=12, then hand-consolidated). Deliberately small: 8 actionable intents + `other`.
Rough keyword-based distribution over the corpus is shown for scale (not the final
labels — those come from the golden set).

| intent | definition | ~share | router risk |
|--------|------------|-------:|:-----------:|
| `flight_disruption` | Delay, cancellation, diversion, missed connection, IRROPS, stranded/rebooked (weather/mechanical). | ~13% | high |
| `baggage` | Lost / delayed / damaged bags; baggage fees, allowance, policy. | ~6% | high |
| `booking_reservation` | Any booking/fare matter: new reservations, schedules, routes, prices, changes, cancellations, refunds, name corrections. | ~10% | high |
| `seat_upgrade` | Seat assignment/change, upgrade lists, cabin (Comfort+, First, Premium Select), standby. | ~10% | high |
| `checkin_boarding` | Check-in problems, boarding passes, gate/boarding issues, kiosk/app/website errors during travel. | ~10% | medium |
| `loyalty_miles` | SkyMiles, Medallion status, miles/points not posting, Sky Club, companion certs, co-brand card benefits. | ~6% | high |
| `complaint` | General dissatisfaction with service/staff or an unresolved prior issue — no specific actionable request. | ~4% | medium |
| `compliment` | Praise for staff, crew, or the airline. | ~11% | low |
| `other` | Off-topic, spam, press/news, not addressed to support. | ~30% | low |

## Why this shape

- **8 + other**, not 20: the golden set is only 150–250 examples; more classes than
  this makes per-intent metrics too thin to trust.
- **`booking_reservation` is one bucket**, not split into "info request" vs "change my
  booking". The *intent* is the same topic; what actually drives auto-vs-escalate is
  whether the customer references an existing itinerary/PNR — that's a **router signal**,
  not a separate intent.
- **`complaint` vs `compliment` are separate** even though both are "feedback": they get
  opposite treatment (compliment → auto thank-you; complaint → human).
- **`other` is large and expected** on Twitter (hashtag chatter, news, brand mentions).
  Routing it correctly (never auto-reply to spam) matters more than sub-classifying it.

## Risk class

Each intent carries a coarse risk label used by the router. `high` = resolving it
needs account/PNR-level action a public tweet can't safely do, so the default is
escalate unless the reply is purely informational.
