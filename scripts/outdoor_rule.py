#!/usr/bin/env python3
"""Decide outdoor vs indoor for a clip, from the caption the VLM already wrote.

There is no indoor/outdoor field in the annotation schema, and adding one means
re-annotating the corpus through the caption API. The dense caption already opens by
stating the scene ("An indoor lounge area features…", "A city street is lined with…"),
so the split can be read out of text we have paid for once.

Tuned for PRECISION of the outdoor set rather than recall: among `real_world` clips about
85% are outdoor already, so the job is to remove the ~15% that are not, and the candidate
pool is larger than any batch cut from it. A caption carrying explicit indoor evidence is
dropped even when outdoor words also appear — "a mix of indoor and outdoor spaces" is
exactly the clip an outdoor batch must not contain.

MEASURED on 50 hand-labelled captions (25 miradata + 25 sekai_walking, frozen by clip id
because the annotation directory grows under a reseeded sample):

    precision 97.1%   recall 81.0%   indoor leaked 0/6   ambiguous kept 1/2

The recall cost is real and is the intended trade: the eight misses are open-air clips
whose captions name only one outdoor thing, or a covered sidewalk whose caption says
"ceiling". Do not tune these lists against that same 50 — it is the only labelled set
there is, and a rule fitted to it would report a precision it does not have.
"""

# Words that only make sense under a roof. Shopping malls, exhibition halls and covered
# markets are the recurring false positives in city-walking footage: they are full of
# storefronts, signage and street furniture, and read as streets to a word count.
INDOOR = (
    "indoor", "indoors", "interior", "ceiling", "atrium", "lobby", "hallway", "corridor",
    "showroom", "warehouse", "shopping mall", "exhibition hall", "market hall",
    "concourse", "foyer", "vestibule", "inside the building", "escalator",
    "living room", "bedroom", "bathroom", "kitchen", "office space", "classroom",
)

# Open-air evidence. Two DISTINCT terms are required, so one incidental "sky" (a skylight,
# a window view) cannot carry a clip on its own.
OUTDOOR = (
    "sky", "clouds", "sunlight", "daylight", "street", "streets", "road", "roadway",
    "sidewalk", "pavement", "outdoor", "outdoors", "crosswalk", "alley", "plaza",
    "courtyard", "boardwalk", "harbor", "harbour", "waterfront", "park", "forest",
    "woods", "trail", "path", "river", "lake", "ocean", "beach", "desert", "valley",
    "mountain", "hillside", "snow-covered", "trees", "vegetation", "foliage", "grass",
    "facade", "facades", "storefronts", "balconies", "rooftop", "skyline", "horizon",
    "parking lot", "bridge", "canal", "field",
)


def classify(caption: str) -> str:
    """-> 'outdoor' | 'indoor' | 'unclear'."""
    c = " ".join((caption or "").lower().split())
    if any(w in c for w in INDOOR):
        return "indoor"
    return "outdoor" if len({w for w in OUTDOOR if w in c}) >= 2 else "unclear"
