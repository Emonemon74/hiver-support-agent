"""Human review layer for the golden set.

`prelabel.py` produced model suggestions. This module encodes the reviewer's
final decisions on top of them, following a written rubric:

ROUTE = "escalate" if ANY of:
  E-ACCOUNT   : a correct answer needs account / PNR / ticket / bag-file lookup,
                or an actual change to the reservation. ("Please DM your
                confirmation number" IS escalate — the resolution happens in a
                human-staffed DM with account access.)
  E-MONEY     : a refund / credit / compensation / fee dispute is at stake.
  E-SAFETY    : safety, medical, assault, discrimination, bereavement, legal, or
                emotional-distress claim.
  E-DISRUPTION: an active disruption where the customer needs rebooking or
                connection help now.
  E-CHURN     : an angry customer explicitly threatening to leave / go public /
                escalate formally, who needs personal handling.
  E-LIVEDATA  : answering correctly needs live flight status / schedule / seat
                availability the agent cannot verify.
otherwise "auto":
  A-COMPLIMENT: praise — safe to thank publicly and forward internally.
  A-POLICY    : a general policy / product question answerable from precedent.
  A-FEEDBACK  : feedback or venting with no actionable request.
  A-MINOR     : acknowledgement of a minor, already-resolving situation.

`difficulty="ambiguous"` marks rows where a second reasonable labeller could
easily disagree (kept for slice metrics and the failure analysis).
"""
from __future__ import annotations

REASONS = {
    "E-ACCOUNT": "Needs account/PNR lookup or a change to the booking; resolved by a human in DMs.",
    "E-MONEY": "A refund, credit or fee dispute is at stake — needs a human decision.",
    "E-SAFETY": "Safety/medical/legal/bereavement issue — must be handled by a person.",
    "E-DISRUPTION": "Active disruption; the customer needs rebooking or connection help now.",
    "E-CHURN": "Angry customer at churn risk or going public — needs personal handling.",
    "E-LIVEDATA": "Answering correctly needs live flight/schedule/seat data the agent can't verify.",
    "A-COMPLIMENT": "Compliment — safe to thank publicly and forward internally.",
    "A-POLICY": "General policy/product question answerable from past replies.",
    "A-FEEDBACK": "Feedback or venting with no actionable request.",
    "A-MINOR": "Acknowledgement of a minor, already-resolving situation.",
}

# thread_id -> trigger code (final route is 'escalate' iff code starts with 'E-')
TRIGGER: dict[int, str] = {
    102272: "E-ACCOUNT", 104250: "A-COMPLIMENT", 1516207: "A-MINOR", 106513: "A-COMPLIMENT",
    446151: "E-ACCOUNT", 446147: "A-COMPLIMENT", 1958351: "A-COMPLIMENT", 2021076: "A-COMPLIMENT",
    1495498: "A-FEEDBACK", 448938: "E-ACCOUNT", 448948: "E-ACCOUNT", 448908: "A-POLICY",
    2657371: "A-COMPLIMENT", 800536: "E-ACCOUNT", 448903: "A-COMPLIMENT", 2657262: "A-COMPLIMENT",
    450925: "A-FEEDBACK", 450931: "A-COMPLIMENT", 450929: "A-FEEDBACK", 469584: "A-COMPLIMENT",
    452187: "E-ACCOUNT", 452148: "E-MONEY", 453555: "E-LIVEDATA", 453552: "E-ACCOUNT",
    454752: "A-FEEDBACK", 454759: "A-COMPLIMENT", 454745: "A-FEEDBACK", 455661: "E-ACCOUNT",
    455666: "A-POLICY", 2119026: "E-ACCOUNT", 456879: "E-CHURN", 456887: "E-ACCOUNT",
    466133: "E-SAFETY", 2880143: "A-COMPLIMENT", 458135: "A-MINOR", 458118: "A-FEEDBACK",
    458096: "E-LIVEDATA", 459177: "E-ACCOUNT", 459175: "E-ACCOUNT", 459165: "E-ACCOUNT",
    459162: "A-COMPLIMENT", 460314: "A-FEEDBACK", 462705: "A-FEEDBACK", 470298: "A-COMPLIMENT",
    463612: "E-DISRUPTION", 1751270: "A-COMPLIMENT", 1953883: "A-COMPLIMENT", 466150: "A-POLICY",
    467628: "A-FEEDBACK", 488943: "A-COMPLIMENT", 471670: "E-ACCOUNT", 471667: "A-FEEDBACK",
    472587: "E-MONEY", 478226: "E-ACCOUNT", 478216: "A-COMPLIMENT", 478209: "E-ACCOUNT",
    479964: "A-COMPLIMENT", 479947: "E-ACCOUNT", 492538: "A-COMPLIMENT", 485595: "E-DISRUPTION",
    485588: "A-MINOR", 511870: "A-POLICY", 487010: "E-DISRUPTION", 487020: "E-MONEY",
    490877: "E-ACCOUNT", 490825: "A-FEEDBACK", 492523: "E-ACCOUNT", 492519: "A-COMPLIMENT",
    492501: "A-FEEDBACK", 494340: "E-ACCOUNT", 494337: "A-COMPLIMENT", 496228: "E-DISRUPTION",
    496219: "A-FEEDBACK", 496215: "E-LIVEDATA", 496198: "A-POLICY", 497833: "A-COMPLIMENT",
    497828: "A-POLICY", 497808: "A-COMPLIMENT", 497819: "E-ACCOUNT", 499381: "E-ACCOUNT",
    501069: "E-LIVEDATA", 502740: "A-POLICY", 504376: "E-SAFETY", 504418: "A-FEEDBACK",
    504407: "E-ACCOUNT", 502757: "E-ACCOUNT", 506148: "E-CHURN", 507530: "E-ACCOUNT",
    506130: "A-FEEDBACK", 507521: "A-COMPLIMENT", 511886: "E-ACCOUNT", 510620: "A-FEEDBACK",
    510612: "A-POLICY", 511880: "A-FEEDBACK", 511767: "A-COMPLIMENT", 511862: "A-MINOR",
    513119: "A-FEEDBACK", 513078: "A-COMPLIMENT", 514188: "A-POLICY", 515518: "A-FEEDBACK",
    514182: "E-ACCOUNT", 515510: "E-LIVEDATA", 514160: "A-COMPLIMENT", 515495: "E-ACCOUNT",
    515478: "A-POLICY", 516828: "E-ACCOUNT", 517822: "E-ACCOUNT", 517807: "A-COMPLIMENT",
    517795: "A-COMPLIMENT", 518994: "A-FEEDBACK", 518971: "E-ACCOUNT", 519948: "A-FEEDBACK",
    520886: "A-FEEDBACK", 521832: "E-LIVEDATA", 522772: "A-FEEDBACK", 523632: "E-MONEY",
    524372: "E-ACCOUNT", 527853: "E-DISRUPTION", 534636: "A-FEEDBACK", 535335: "E-ACCOUNT",
    540393: "E-DISRUPTION", 540401: "A-COMPLIMENT", 540397: "E-DISRUPTION", 540386: "E-DISRUPTION",
    541686: "E-MONEY", 570547: "E-SAFETY", 545972: "E-SAFETY", 547223: "E-ACCOUNT",
    547226: "E-DISRUPTION", 548715: "A-FEEDBACK", 550075: "E-DISRUPTION", 550070: "A-COMPLIMENT",
    550073: "A-MINOR", 552721: "A-FEEDBACK", 552705: "A-COMPLIMENT", 554206: "A-POLICY",
    554192: "A-FEEDBACK", 558661: "A-FEEDBACK", 558654: "E-MONEY", 558650: "A-POLICY",
    561050: "E-MONEY", 560049: "A-POLICY", 560031: "A-POLICY", 561025: "A-FEEDBACK",
    563564: "E-ACCOUNT", 562479: "E-DISRUPTION", 562475: "A-COMPLIMENT", 563551: "A-POLICY",
    564595: "A-COMPLIMENT", 564586: "A-COMPLIMENT", 565486: "A-FEEDBACK", 565483: "A-FEEDBACK",
    566410: "E-ACCOUNT", 567243: "E-ACCOUNT", 566382: "E-ACCOUNT", 567226: "E-CHURN",
    567224: "A-COMPLIMENT", 567230: "A-MINOR", 571406: "A-POLICY", 572712: "E-MONEY",
    573255: "A-POLICY", 573253: "E-MONEY", 573801: "A-COMPLIMENT", 574730: "E-CHURN",
    575157: "E-ACCOUNT", 575160: "A-POLICY", 576244: "E-LIVEDATA", 576817: "E-ACCOUNT",
    579527: "A-FEEDBACK", 581587: "E-ACCOUNT", 582316: "A-COMPLIMENT", 585243: "A-POLICY",
    584272: "A-FEEDBACK", 584269: "E-LIVEDATA", 585238: "A-COMPLIMENT", 587578: "A-FEEDBACK",
    587576: "A-COMPLIMENT", 587561: "E-SAFETY", 2511074: "E-MONEY", 590202: "E-ACCOUNT",
    591647: "E-DISRUPTION", 594138: "E-ACCOUNT", 595252: "E-ACCOUNT", 595250: "E-ACCOUNT",
    595244: "E-DISRUPTION", 596364: "E-MONEY", 597410: "A-POLICY", 597441: "E-LIVEDATA",
    597387: "A-FEEDBACK", 598574: "E-ACCOUNT", 599513: "A-COMPLIMENT", 600922: "A-FEEDBACK",
    601903: "E-ACCOUNT", 601886: "E-DISRUPTION", 602877: "A-FEEDBACK", 603668: "A-FEEDBACK",
    603660: "A-COMPLIMENT", 603641: "A-COMPLIMENT", 603625: "A-POLICY",
}

# reviewer overrides to the model's intent
INTENT_FIX: dict[int, str] = {
    502757: "booking_reservation",
    563564: "complaint",
}

# a second reasonable labeller could disagree on route and/or intent for these
AMBIGUOUS: set[int] = {
    1495498, 800536, 453555, 455661, 458096, 467628, 471667, 490877, 492538, 496215,
    510612, 511862, 521832, 550073, 552705, 565483, 565486, 571406, 575157, 579527,
    563551, 585243, 597410, 598574, 602877, 603668, 506148, 2511074,
}
