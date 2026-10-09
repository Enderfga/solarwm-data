"""Full-video H3 caption contract: retain content, exclude control leakage.

Lexical guards catch explicit camera/control prose. They do not decide whether
an actor is the principal subject, or whether movement is real rather than
parallax. Those are visual/semantic review questions, not an action-word ban.
"""
from __future__ import annotations

import copy
import hashlib
import re
from collections.abc import Mapping
from typing import Any

SCHEMA_VERSION = "solarwm.h3_i2va.caption_pilot.v3"
JSON_ENVELOPE_POLICY = "strict_json_envelope_v1"
AUDIO_POLICY = "video_only_silent"
MOTION_TEXT_POLICY = "exclude_camera_and_principal_subject_motion_preserve_background_dynamics"
PERSPECTIVE_TYPES = ("first_person", "third_person", "uncertain")
VISUAL_STYLES = (
    "photographic live action",
    "three-dimensional computer-generated imagery",
    "two-dimensional animation",
    "mixed or uncertain visual medium",
)
FACT_KEYS = frozenset({"perspective_type", "visual_style", "scene_description", "scene_dynamics"})
FIRST_FRAME_INSTRUCTION = (
    "For the target video, at 0.00 seconds into the target video, "
    "<Picture 1> (from [Shot 1]) is fully referenced."
)
TEXT_LIMITS = {"visual_style": (1, 160), "scene_description": (1, 5000), "scene_dynamics": (0, 2000)}
CAMERA_GUARDS = (
    ("camera_motion", r"\b(?:camera|viewpoint|perspective|viewing position|field of view|framing|our view|the view)\b(?:(?!\b(?:and|while|whereas|but)\b)[^.!?;,]){0,70}\b(?:moves?|moving|pans?|panning|tilts?|tilting|zooms?|zooming|doll(?:y|ies|ying)|orbits?|orbiting|tracks?|tracking|follows?|following|rotates?|rotating|advances?|advancing|retreats?|retreating|sweeps?|sweeping|drifts?|drifting|shifts?|shifting)\b"),
    ("camera_setup", r"\b(?:camera|viewpoint|viewing position)\s+(?:is\s+|remains\s+|stays\s+)?(?:static|fixed|stationary|handheld|eye[- ]level)\b"),
    ("cinematic_instruction", r"\b(?:zoom(?:s|ed|ing)?\s+(?:in|out)|dolly\s+(?:in|out)|pan(?:s|ned|ning)?\s+(?:left|right|across)|tilt(?:s|ed|ing)?\s+(?:up|down)|(?:tracking|establishing|aerial|close[- ]up|wide|medium)\s+shot|fly[- ]?through|ego[- ]?motion)\b"),
    ("observer_movement", r"\b(?:we|the viewer|the observer)\s+(?:(?:slowly|gradually|then)\s+)*(?:move\w*|approach\w*|advance\w*|retreat\w*|turn\w*|walk\w*|travel\w*)\b"),
    ("viewing_relation", r"\b(?:viewed|seen|observed)\s+(?:from|through)\b|\b(?:toward|towards|away from|past|facing)\s+(?:the\s+)?(?:camera|viewer|observer)\b"),
    ("rear_observer_view", r"\b(?:visible|shown)\s+from\s+behind(?=\s+(?:throughout|in|during|across|over)\b|[.!?;,]|$)"),
    ("back_to_observer", r"\bback\s+(?:is\s+)?(?:to|toward|towards|facing)\s+(?:the\s+)?(?:camera|viewer|observer)\b"),
    ("parallax_as_dynamics", r"\b(?:the\s+)?(?:background|surroundings|entire scene|whole scene|landscape|environment)\s+(?:(?:slowly|gradually|steadily|smoothly)\s+)*(?:slides?|sliding|drifts?|drifting|sweeps?|sweeping|shifts?|shifting)\b"),
    ("frame_chronology", r"\b(?:in|during|by|across|throughout)\s+(?:the\s+)?(?:some|several|certain|earlier|later|early|subsequent|following|successive|remaining|multiple|all|any|each|every|last|final|initial|first)\s+(?:frames?|views?|images?|observations?)\b|\b(?:as|while)\s+(?:the\s+)?(?:sequence|clip|video|footage)\s+(?:progresses|advances|continues)\b"),
    ("screen_trajectory", r"\b(?:enters?|exits?|leaves?)\s+(?:from\s+)?(?:the\s+)?(?:frame|view|screen)\b|\b(?:left[- ]to[- ]right|right[- ]to[- ]left|screen[- ]space|screen[- ]relative)\b"),
)
PRINCIPAL_GUARDS = (
    ("explicit_principal_action", r"\b(?:main|principal|central)\s+(?:subject|character|person|figure|animal|vehicle|bird|dog|cat|horse|cow|man|woman|child|pedestrian|cyclist|driver)\s+(?:(?:is|are|then|slowly|quickly|continues? to)\s+)*(?:walk\w*|run\w*|turn\w*|mov\w*|jump\w*|gestur\w*|wav\w*|danc\w*|driv\w*|rid\w*|swim\w*|fly\w*|eat\w*|graz\w*|nod\w*|reach\w*|rais\w*|lower\w*|step\w*|flap\w*|hop\w*|leap\w*|climb\w*|pedal\w*|kick\w*|shak\w*|bend\w*|stretch\w*|chew\w*|drink\w*)\b"),
)
OTHER_GUARDS = (
    ("speculative_wording", r"\b(?:suggesting|suggests?|seems?|apparently|possibly|probably|likely|perhaps)\b"),
    ("audio_claim", r"\b(?:audible|audibly|soundscape|soundtrack|dialogue|voice[- ]?over|narration|silence|silent|heard|hears|hearing)\b"),
    ("music_claim", r"(?<!sheet-)(?<!sheet )\bmusic\b(?!\s+(?:shops?|stores?|schools?|books?|scores?|stands?|posters?|venues?|rooms?)\b)"),
    ("template_injection", r"(?:integrated_multimodal_description|overall_soundscape|non_diegetic_music)\s*:|\[Shot\s+\d+\]|<Picture\s+\d+>|```"),
    ("perspective_label_in_prose", r"\b(?:first_person|third_person|perspective_type)\b|\b(?:first|third)[- ]person\s+(?:view|perspective)\b"),
)


# Explicit output-policy violations observed in raw production canaries.
# These guards do not infer visual roles or decide whether motion is parallax.
VALIDATOR_POLICY_VERSION = "h3_caption_explicit_leakage_v8"
CAMERA_GUARDS += (
    ("view_changes", r"\b(?:as|while)\s+(?:the\s+)?(?:view|viewpoint|camera|observer)\s+changes?\b"),
    ("observer_possessive_viewpoint", r"\b(?:rider|driver|observer|viewer)(?:['’]s|s['’])\s+(?:view|viewpoint|perspective)\b"),
    ("later_reveal", r"\b(?:and|with|then)\s+(?:later|earlier)\s+(?:a|an|the)\b"),
)
OTHER_GUARDS += (
    ("perspective_reasoning", r"\b(?:distinct\s+from|other\s+than)\s+(?:the\s+)?(?:principal|main)\s+(?:figure|subject|character)\b"),
)

def _explicit_dynamic_policy(value: Any) -> list[str]:
    if not isinstance(value, dict):
        return []
    errors=[]
    dynamics=value.get("scene_dynamics")
    description=value.get("scene_description")
    if isinstance(dynamics,str) and dynamics:
        for match in re.finditer(r"\b(?:remain|remains|stay|stays|are|is)\s+(?:still|motionless|stationary|unchanging)\b",dynamics,re.I):
            errors.append(f"scene_dynamics: unsupported_output:static_absence; matched {match.group(0)!r}. Put supported static facts in scene_description; scene_dynamics contains independent background events only.")
    if value.get('perspective_type')=='third_person' and isinstance(description,str):
        # The first grammatical person subject is the declared external principal.
        # Restrict to its opening sentence; later/background people remain allowed.
        opening=description.split('.',1)[0]
        subject=re.match(r"(?:A|An|The)\s+(?:man|woman|person|figure|character|boy|girl|soldier|knight)\b",opening,re.I)
        if subject:
            action=re.search(r"\b(?:stands|sits|kneels|crouches)\s+and\s+(?:walks|runs|jumps|gestures|dances|nods|kicks|bends|stretches)\b",opening[subject.end():],re.I)
            # Avoid an embedded second subject or an explicit background clause.
            if action and not re.search(r"\b(?:while|whereas|another|other|background|people|pedestrians|and\s+(?:a|an|the)\s+(?:man|woman|person|figure|character))\b",opening[subject.end():subject.end()+action.start()],re.I):
                errors.append(f"scene_description: principal_subject_motion:opening_principal_action; matched {action.group(0)!r}. Preserve the declared principal's appearance and supported posture, omit its action, and retain independent background activity.")
    return errors


CAMERA_GUARDS += (
    ("sampled_image_narration", r"\b(?:across|between|throughout|over|in|at)\s+(?:the\s+)?frames?\b|\b(?:early|later|final|initial|first|last)\s+(?:portion|part|sections?)\s+(?:of\s+)?(?:the\s+)?(?:sequence|clip|video)\b|\b(?:in|during)\s+(?:one|a single)\s+frame\b|\bat\s+one\s+point\s+in\s+(?:the\s+)?sequence\b"),
    ("frame_position", r"\b(?:upper|lower|top|bottom)[- ](?:left|right)\s+corner\b|\b(?:foreground|background|edges?|corners?)\s+of\s+(?:the\s+)?(?:some\s+)?(?:views?|frames?|images?)\b"),
    ("observer_reveal", r"\b(?:becomes?|became|becoming)\s+visible\b"),
    ("aerial_view_prose", r"\baerial\s+views?\b"),
)
OTHER_GUARDS += (
    ("overlay_or_optical_artifact", r"\b(?:HUD|heads?[- ]up\s+display|subtitles?|watermarks?|lens\s+flares?|double\s+exposure|on[- ]screen\s+interface|interface\s+(?:elements?|overlay)|(?:digital|navigation|telemetry)\s+(?:navigation\s+)?overlay)\b"),
)

_declared_entity = r"(?:man|woman|person|figure|character|boy|girl|soldier|knight|men|women|people|horse|dog|cat|animal|car|vehicle|motorcycle|boat|motorboat|cyclist|biker|mountain biker|rider)"
_principal_finite_action = r"(?:walks?|runs?|jumps?|gestures?|dances?|nods?|kicks?|bends?|stretches?|moves?|travels?|drives?|gallops?|deploys?|attacks?|fights?|swings?|climbs?|swims?|turns?|rotates?|leaps?|rides?|pedals?)"

def _declared_principal_policy(value: Any) -> list[str]:
    if not isinstance(value,dict):return []
    errors=[]
    dynamics=value.get('scene_dynamics')
    if isinstance(dynamics,str):
        match=re.search(r"\b(?:maintains?|retains?)\s+(?:their|its)\s+(?:illumination|glow)\b",dynamics,re.I)
        if match:errors.append(f"scene_dynamics: unsupported_output:static_illumination; matched {match.group(0)!r}. Move supported steady illumination to scene_description; retain independent background events.")
    if value.get('perspective_type')!='third_person':return errors
    description=value.get('scene_description')
    if not isinstance(description,str):return errors
    for index,sentence in enumerate(re.split(r'[.!?]',description)):
        sentence=sentence.strip()
        # Opening declared subject, or explicit subsequent reference to it.
        start=re.match(r'(?:A|An|The|Two|Three)\s+(?:[\w-]+\s+){0,6}?'+_declared_entity+r'\b',sentence,re.I) if index==0 else re.match(r'The\s+(?:principal\s+|main\s+)?(?:subject|character|figure|person)\b',sentence,re.I)
        if not start:continue
        rest=sentence[start.end():]
        match=re.search(r'\b'+_principal_finite_action+r'\b',rest,re.I)
        if not match:continue
        before=rest[:match.start()]
        if re.search(r'\b(?:another|other|background|pedestrians|villagers|people|while|whereas|where|that|which)\b|\band\s+(?:a|an|the)\s+'+_declared_entity+r'\b',before,re.I):continue
        # "wooden steps" and "runs of stone" are deliberately outside the verbs.
        if match.group(0).lower() in ('run','runs') and re.match(r'\s+of\b',rest[match.end():],re.I):continue
        errors.append(f"scene_description: principal_subject_motion:declared_subject_action; matched {match.group(0)!r}. Preserve the declared principal's supported appearance and posture; omit action and travel control, while retaining independent background motion.")
    return errors


# These expressions recognize explicit output categories, not visual roles.
CAMERA_GUARDS += (
    ("article_frame_narration", r"\b(?:in|during)\s+(?:an?|the)\s+(?:early|later|initial|final|first|last|opening|ending)\s+frames?\b"),
    ("sequence_presence_narration", r"\b(?:is|are)\s+(?:present|visible)\s+throughout\s+(?:(?:a|an|the)\s+)?(?:sequence|clip|video|footage|observations?|frames?)\b|\bappears?\s+repeatedly\b|\b(?:early|later|initial|final|opening|ending)\s+(?:sections?|portions?)\b"),
    ("image_direction", r"\b(?:upper|lower|top|bottom)[- ](?:left|right)\b|\b(?:left|right)\s+edge(?=\s+(?:of\s+(?:the\s+)?(?:frame|view|image|screen)|[.!?;,])|[.!?;,]|$)"),
    ("observer_body_controls", r"\b(?:observer|viewer|driver|rider)(?:['’]s|s['’])\s+(?:hands?|arms?|feet)\b|\ba\s+hand\b[^.!?;]{0,75}\b(?:controls?|steering\s+wheel|handlebars?)\b"),
)
PRINCIPAL_GUARDS += (
    ("principal_or_observer_shadow_motion", r"\b(?:(?:principal|main|central)\s+)?(?:subject|figure|character|person|man|woman|rider|driver|cyclist|biker|observer|viewer)(?:['’]s|s['’])\s+shadow\b[^.!?;]{0,100}\b(?:moves?|moving|shifts?|shifting|changes?|changing|stretches?|stretching)\b"),
)

def _reviewed_v3_policy(value: Any) -> list[str]:
    if not isinstance(value,dict):return []
    dynamics=value.get('scene_dynamics')
    if not isinstance(dynamics,str) or not dynamics:return []
    errors=[]
    # Keep BG "stands up", "sits down", "stands and gestures", and
    # "sits beside a table, raising an arm" as supported physical events.
    for sentence in re.split(r'[.!?;]',dynamics):
        for match in re.finditer(r"\b(?:stands?|sits?|kneels?|crouches?)\s+(?:near|beside|by|at|on|in|with|around|still|motionless)\b",sentence,re.I):
            rest=sentence[match.end():]
            rest=re.split(r"\b(?:while|whereas|another|other)\b|\band\s+(?:(?:a|an|the|two|three|several)\s+)?(?:man|woman|person|figure|people|pedestrians|men|women)\b",rest,maxsplit=1,flags=re.I)[0]
            dynamic_continuation=re.search(r"(?:\band\s+(?:(?:then|also|occasionally)\s+)*|,\s*(?:(?:one|both|each)\s+)?)(?:gestur\w*|rais\w*|lower\w*|wav\w*|mov\w*|shift\w*|turn\w*|reach\w*|bend\w*|nod\w*|walk\w*|interact\w*|handl\w*)\b",rest,re.I)
            if not dynamic_continuation:
                errors.append(f"scene_dynamics: unsupported_output:static_posture; matched {match.group(0)!r}. Move supported static posture to scene_description; preserve clearly observed independent background actions, physical flames, smoke, water and rain.")
        for match in re.finditer(r"\b(?:remain|remains|stay|stays)\s+(?:seated|standing|near|beside|at|in\s+(?:static\s+)?positions?|red|green|illuminated)\b|\b(?:maintain|maintains)\s+(?:raised\s+arm\s+)?postures?\b|\b(?:is|are)\s+present\b",sentence,re.I):
            # A changing traffic state with an explicit transition is physical BG dynamics.
            if re.search(r"\b(?:until|before|then)\b[^.!?;]{0,70}\b(?:changes?|turns?|switches?)\b",sentence[match.end():],re.I):continue
            errors.append(f"scene_dynamics: unsupported_output:static_presence; matched {match.group(0)!r}. Keep supported static presence/illumination in scene_description, not scene_dynamics; retain independent background events.")
    return errors


CAMERA_GUARDS += (
    ("explicit_observation_chronology", r"\b(?:across|between|over|throughout|in)\s+(?:the\s+)?observations\b|\b(?:in|during)\s+(?:the\s+)?(?:early|earlier|later|final|initial)\s+screens\b|\b(?:in|during)\s+(?:one|a|another)\s+(?:part|portion)\s+of\s+(?:the\s+)?(?:sequence|video|clip|footage)\b|\b(?:visible|appears?|seen)\s+at\s+one\s+point\b"),
)
OTHER_GUARDS += (
    ("unestablished_object_hedge", r"\bwhat\s+appears\s+to\s+be\b"),
    ("floating_interface_marker", r"\b(?:translucent|transparent)\s+(?:(?:square|rectangular|circular)\s+)?(?:frame\s+)?marker\b[^.!?;]{0,60}\b(?:floats?|floating)\s+in\s+(?:the\s+)?air\b|\bfloating\s+(?:interface|targeting|reticle)\s+(?:marker|element|overlay)\b"),
)

def _reviewed_v4_policy(value: Any) -> list[str]:
    if not isinstance(value,dict):return []
    errors=[]
    description=value.get('scene_description');dynamics=value.get('scene_dynamics')
    if isinstance(dynamics,str) and dynamics:
        for sentence in re.split(r'[.!?]',dynamics):
            for match in re.finditer(r"\b(?:stands?|standing|sits?|sitting)\s+or\s+(?:walk\w*|mov\w*)\b",sentence,re.I):
                errors.append(f"scene_dynamics: unsupported_output:uncommitted_posture_or_motion; matched {match.group(0)!r}. State only established independent physical events; static posture belongs in scene_description.")
            for match in re.finditer(r"\b(?:remains?|stays?)\s+in\s+conversation\b|\b(?:is|are|remains?|stays?)\s+(?:also\s+)?visible\b",sentence,re.I):
                continuation=sentence[match.end():]
                if re.match(r"\s+in\s+traffic\b",continuation,re.I) and re.search(r"\bVehicles\s+(?:move|travel|pass)\b",sentence[:match.start()],re.I):continue
                if re.search(r"\b(?:gestur\w*|nod\w*|wav\w*|mouth\w*|lips|mov\w*|walk\w*|run\w*|swim\w*|dart\w*|flap\w*|fly\w*|flying|rais\w*|lower\w*)\b",continuation,re.I):continue
                errors.append(f"scene_dynamics: unsupported_output:static_presence; matched {match.group(0)!r}. Move supported static presence to scene_description, and preserve explicitly observed independent gestures, interactions and other events.")
    if isinstance(description,str) and value.get('perspective_type')=='first_person':
        match=re.search(r"\b(?:a|the)\s+hand\b(?:(?!\b(?:sculpted|statue|sculpture|mannequin|painting|person|woman|man|background)\b)[^.!?;]){0,100}\bbriefly\s+(?:appears?|is\s+visible)\b",description,re.I)
        if match:errors.append(f"scene_description: camera_control:unbound_observer_hand; matched {match.group(0)!r}. Omit observer body/control narration; preserve supported physical environment and independent background actors.")
    if value.get('perspective_type')!='third_person' or not isinstance(description,str):return errors
    for field,text in [('scene_description',description),('scene_dynamics',dynamics)]:
        if not isinstance(text,str):continue
        for sentence in re.split(r'[.!?;]',text):
            match=re.search(r"\b(?:the\s+)?(?:principal|main|central)\s+(?:subject|figure|character|person)(?:['’]s)\s+(?:[\w-]+\s+){0,6}?(?:coat|cloak|robe|garment|clothing|lining|fabric|edges)\b[^.!?;]{0,100}\b(?:shifts?|moves?|sways?|flutters?)\b[^.!?;]{0,60}\b(?:with|as|during)\b[^.!?;]{0,35}\b(?:movement|motion|moves?|walks?)\b",sentence,re.I)
            if match:errors.append(f"{field}: principal_subject_motion:explicit_principal_garment_movement; matched {match.group(0)!r}. Preserve principal garment appearance without its movement-caused changes; retain independent background dynamics.")
            if field=='scene_description':
                match=re.search(r"\b(?:is|are)\s+(?:present|visible)\s+throughout\s*$",sentence,re.I)
                if match and re.search(r"\b(?:he|she|person|figure|character|man|woman|subject)\b",sentence[:match.start()],re.I) and not re.search(r"\b(?:another|other|background)\b",sentence[:match.start()],re.I):
                    errors.append(f"{field}: camera_control:unqualified_subject_persistence; matched {match.group(0)!r}. Keep supported subject appearance without narration of its persistence across observations; explicit physical spatial distribution remains allowed.")
    opening=re.split(r'[.!?]',description)[0]
    declared=re.match(r'(?:A|An|The|Two|Three)\s+(?:[\w-]+\s+){0,6}?'+_declared_entity+r'\b',opening,re.I)
    if not declared:return errors
    for field,text in [('scene_description',description),('scene_dynamics',dynamics)]:
        if not isinstance(text,str):continue
        for sentence in re.split(r'[.!?;]',text):
            for match in re.finditer(r"\b(?:he|she)\s+(?:(?:slowly|gradually|quickly|then|continues?\s+to|is)\s+)*(?:moves?|moving|walks?|walking|runs?|running|travels?|traveling|rides?|riding|swims?|swimming|turns?|turning|gestures?|gesturing)\b",sentence,re.I):
                if re.search(r"\b(?:another|other|background|second|third)\b",sentence[:match.start()],re.I):continue
                errors.append(f"{field}: principal_subject_motion:declared_principal_pronoun; matched {match.group(0)!r}. Keep the declared principal's appearance and supported static posture, omit action/travel, and retain separate independent background events.")
            for match in re.finditer(r"\b(?:his|her|their)\s+(?:[\w-]+\s+){0,6}?(?:coat|cloak|robe|garment|clothing|lining|fabric|edges)\b[^.!?;]{0,100}\b(?:shifts?|moves?|sways?|flutters?)\b[^.!?;]{0,60}\b(?:with|as|during)\b[^.!?;]{0,35}\b(?:movement|motion|moves?|walks?)\b",sentence,re.I):
                if re.search(r"\b(?:another|other|background|second|third)\b",sentence[:match.start()],re.I):continue
                errors.append(f"{field}: principal_subject_motion:principal_garment_movement; matched {match.group(0)!r}. Keep garment appearance; omit its change caused by principal movement, while retaining independent background dynamics.")
            if field=='scene_description':
                match=re.search(r"\b(?:is|are)\s+(?:present|visible)\s+throughout\s*$",sentence,re.I)
                if match and not re.search(r"\b(?:another|other|background|trees|columns|buildings|houses|lamps|chairs)\b",sentence[:match.start()],re.I):
                    errors.append(f"{field}: camera_control:principal_sequence_persistence; matched {match.group(0)!r}. Describe the principal's supported appearance directly, without unqualified persistence across observations. Physical spatial distribution remains allowed.")
    return errors


CAMERA_GUARDS += (
    ("qualified_frame_chronology", r"\b(?:foreground|background|edges?|corners?)\s+of\s+(?:the\s+)?(?:early|earlier|later|initial|final|first|last)\s+(?:frames?|views?|images?)\b|\b(?:middle|opening|ending)\s+(?:portion|part|section)\s+of\s+(?:the\s+)?(?:sequence|video|clip|footage)\b|\bat\s+different\s+moments\b"),
    ("image_relative_physical_edge", r"\b(?:left|right)\s+physical\s+edge\s+of\s+(?:the\s+)?scene\b"),
    ("foreground_trajectory", r"\b(?:moves?|moving|walks?|walking|runs?|running|travels?|traveling)\b[^.!?;]{0,45}\b(?:through|toward|towards|into|across)\s+(?:the\s+)?foreground\b"),
    ("time_qualified_visibility", r"\bat\s+times\s+(?:appears?|is\s+visible)\b"),
    ("viewport_border", r"\b(?:rim|edge)\s+of\s+(?:a|an|the)\s+(?:diving\s+)?(?:mask|helmet\s+visor)\b[^.!?;]{0,70}\b(?:borders?|frames?)\s+(?:the\s+)?visible\s+area\b"),
    ("superimposed_image_artifact", r"\bsuperimposed\b|\boverlaid\s+on\s+(?:the\s+)?(?:image|screen|frame|scene)\b|\bicon\b[^.!?;]{0,80}\bupper\s+center\b"),
)
OTHER_GUARDS += (
    ("unestablished_appearance_hedge", r"\b(?:appears?|appearing)\s+to\s+be\b|\bappears?\s+to\s+(?:shift|move|drift|fall|blow)\b"),
    ("uncommitted_static_posture", r"\b(?:sits?|sitting|stands?|standing|kneels?|kneeling|crouches?|crouching)\s+or\s+(?:sits?|sitting|stands?|standing|kneels?|kneeling|crouches?|crouching)\b"),
    ("unestablished_scene_conditions", r"\bnight\s+or\s+(?:during\s+)?(?:heavy\s+)?overcast\b|\bdusk\s+or\s+dawn\b|\bdawn\s+or\s+dusk\b|\bdusk\s+or\s+night\b|\bnight\s+or\s+(?:(?:in|at|during)\s+)?(?:deep\s+)?(?:twilight|dusk|late\s+evening|early\s+morning)\b|\btwilight\s+or\s+pre-storm\b"),
    ("unseen_light_source_inference", r"\bfrom\s+unseen\s+(?:openings?|windows?|doors?|lights?|sources?)\b"),
)

_declared_entity = _declared_entity[:-1] + r"|bird(?:\s+of\s+prey)?|eagle|SUV|van|truck|bus|plane|helicopter)"
_principal_finite_action = _principal_finite_action[:-1] + r"|pulls?|soars?|flaps?|flies|flying)"

def _reviewed_v5_policy(value: Any) -> list[str]:
    if not isinstance(value,dict):return []
    errors=[];description=value.get('scene_description');dynamics=value.get('scene_dynamics')
    if not isinstance(description,str):return errors
    if isinstance(dynamics,str):
        for match in re.finditer(r"\blies?\s+motionless\b|\bhave\s+illuminated\s+brake\s+lights\b",dynamics,re.I):
            errors.append(f"scene_dynamics: unsupported_output:static_state; matched {match.group(0)!r}. Put supported static appearance in scene_description and retain actual independent background events.")
        match=re.search(r"\boccupy\s+different\s+positions\b[^.!?;]{0,70}\bacross\s+(?:the\s+)?sequence\b",dynamics,re.I)
        if match:errors.append(f"scene_dynamics: camera_control:image_position_as_motion; matched {match.group(0)!r}. Changing image position alone is not a physical event. Retain only independently supported changes relative to physical scene structure.")
        # This is a targeted re-inspection requirement, never an assertion that
        # combustion or a still flame proves independent motion.
        if re.search(r"\blit\s+gas\s+fire\b",description,re.I) and not re.search(r"\b(?:fire|flames?)\b",dynamics,re.I):
            errors.append("scene_dynamics: observed_background_event_check:active_gas_fire. Recheck ALL supplied frames for independently changing flame shape. If clearly supported, retain the fire event in scene_dynamics; if ambiguous, omit the unestablished active-fire claim and keep the fireplace appearance. Do not invent flame movement from a still flame or common knowledge.")
    for match in re.finditer(r"\bsky\s+(?:is\s+)?transitioning\b|\blit\s+with\s+shifting\s+colored\s+lights\b",description,re.I):
        errors.append(f"scene_description: unsupported_output:changing_background_behavior; matched {match.group(0)!r}. Static appearance belongs here; put clearly supported physical background changes only in scene_dynamics, and omit exposure or viewpoint changes.")
    if value.get('perspective_type')=='first_person':
        for sentence in re.split(r'[.!?;]',description):
            match=re.search(r"\ba\s+pair\s+of\s+(?:gloved\s+)?hands\b[^.!?;]{0,100}\b(?:lower|bottom)\s+portion\b",sentence,re.I)
            if match and not re.search(r"\b(?:statue|sculpture|painting|mannequin|background|woman|man|person)\b",sentence,re.I):
                errors.append(f"scene_description: camera_control:unbound_observer_hands; matched {match.group(0)!r}. Omit unbound observer body and image-relative cropping; preserve explicitly embodied physical background actors and object appearance.")
    if value.get('perspective_type')!='third_person':return errors
    # The first explicitly declared external actor can follow a setting sentence.
    # Background/other actors and relative clauses do not establish that principal.
    declared=None
    for sentence in re.split(r'[.!?]',description):
        sentence=sentence.strip()
        start=re.match(r'(?:A|An|The|Two|Three)\s+(?:[\w-]+\s+){0,6}?(?:'+_declared_entity+r')\b',sentence,re.I)
        if not start or re.search(r'\b(?:background|another|other|distant|incidental)\b',start.group(0),re.I):continue
        declared=start;rest=sentence[start.end():]
        match=re.search(r'\b'+_principal_finite_action+r'\b',rest,re.I)
        if match:
            before=rest[:match.start()]
            if not re.search(r'\b(?:another|other|background|pedestrians|villagers|people|while|whereas|where|that|which)\b|\band\s+(?:a|an|the)\s+(?:'+_declared_entity+r')\b',before,re.I):
                if not (match.group(0).lower() in ('run','runs') and re.match(r'\s+of\b',rest[match.end():],re.I)):
                    errors.append(f"scene_description: principal_subject_motion:setting_followed_by_principal_action; matched {match.group(0)!r}. Keep the first declared principal's appearance, omit its action, and retain clearly distinct physical background activities.")
        break
    if declared and re.search(r'\b(?:bird|eagle)\b',declared.group(0),re.I):
        for field,text in [('scene_description',description),('scene_dynamics',dynamics)]:
            if not isinstance(text,str):continue
            for sentence in re.split(r'[.!?;]',text):
                match=re.search(r"\b(?:the\s+)?(?:bird|eagle)(?:['’]s)\s+wings?\b[^.!?;]{0,60}\b(?:change|changes|changing|flap|flaps|flapping)\b",sentence,re.I)
                if match and not re.search(r'\b(?:background|another|other)\b',sentence[:match.start()],re.I):
                    errors.append(f"{field}: principal_subject_motion:declared_bird_wing_configuration; matched {match.group(0)!r}. Retain principal plumage/appearance, omit its changing body configuration, and preserve supported distinct background actors.")
    return errors


CAMERA_GUARDS += (
    ("explicit_multiple_frame_extension", r"\bextends?\s+through\s+multiple\s+frames\b"),
    ("explicit_viewing_surface_artifact", r"\b(?:water\s+)?(?:droplets?|streaks?)\b[^.!?;]{0,70}\bon\s+(?:the\s+)?viewing\s+surface\b"),
    ("early_sequence_reveal", r"\bappears?\s+early\s+(?:in\s+)?(?:the\s+)?sequence\b|\bnear\s+(?:the\s+)?start\b(?!\s+of\s+(?!the\s+(?:sequence|video|clip|footage)\b))"),
)

def _reviewed_v6_policy(value: Any) -> list[str]:
    if not isinstance(value,dict):return []
    errors=[];description=value.get('scene_description');dynamics=value.get('scene_dynamics')
    if isinstance(description,str):
        for match in re.finditer(r"\b(?:particles|embers|snow|rain)\s+(?:drift|drifts|fall|falls|descend|descends)\b",description,re.I):
            errors.append(f"scene_description: unsupported_output:particle_behavior_in_static_field; matched {match.group(0)!r}. Preserve supported particle appearance here and independently changing physical behavior only in scene_dynamics. Do not infer movement from a still depiction.")
        for sentence in re.split(r'[.!?;]',description):
            # Bind to an explicitly unowned hand; background actor hands and
            # physical art remain legal in any perspective.
            match=re.search(r"\b(?:a|the)\s+(?:[\w-]+\s+){0,4}?gloved\s+hand\b[^.!?;]{0,100}\b(?:lower|bottom)\s+foreground\b",sentence,re.I)
            if match and not re.search(r"\b(?:statue|sculpture|painting|mannequin|background|woman|man|person|cyclist|rider)(?:['’]s)?\b",sentence[:match.end()],re.I):
                errors.append(f"scene_description: camera_control:unbound_gloved_observer_hand; matched {match.group(0)!r}. Omit unowned observer body and image-relative cropping. Preserve explicitly embodied physical background actors and supported principal appearance.")
            # Changing behavior belongs to dynamics even when the actor is
            # legitimately background. These narrow finite clauses do not ban
            # background dynamics or static posture/appearance.
            for match in re.finditer(r"\b(?:pedestrians|people|women|men|passers|passersby|a\s+man|the\s+man|a\s+woman|the\s+woman)\b[^.!?;]{0,60}\b(?:walks?|examines?|gestures?|raises?|lowers?|waves?|crosses|cross|runs?|reaches?)\b",sentence,re.I):
                after=sentence[match.end():]
                if re.match(r'\s+of\b',after,re.I):continue
                errors.append(f"scene_description: unsupported_output:actor_behavior_in_static_field; matched {match.group(0)!r}. Preserve supported actor appearance here. Move only independently supported background behavior to scene_dynamics; omit principal action/control rather than moving it there.")
    if isinstance(dynamics,str):
        for sentence in re.split(r'[.!?;]',dynamics):
            match=re.search(r"\b(?:people|pedestrians|spectators|a\s+person|the\s+person|a\s+man|a\s+woman|a\s+flag|the\s+flag)\s+(?:sit|sits|stand|stands|hang|hangs)\b",sentence,re.I)
            if not match:continue
            after=sentence[match.end():]
            # Standing up/down and separate subsequent actions are genuine
            # changes; a sitting person conversing can remain background motion.
            if re.match(r'\s+(?:up|down|back\s+up)\b',after,re.I) or re.search(r'\b(?:walk|walks|walking|gesture|gestures|gesturing|converse|converses|conversing|wave|waves|waving|move|moves|moving|rise|rises|raising|turn|turns|turning|shift|shifts|shifting|talk|talks|talking|raise|raises|raising|lower|lowers|lowering|reach|reaches|reaching|bend|bends|bending|nod|nods|nodding)\b',after,re.I):continue
            errors.append(f"scene_dynamics: unsupported_output:static_actor_state; matched {match.group(0)!r}. Keep static posture or flag appearance in scene_description. Retain separately supported physical background changes without inventing motion.")
    return errors


CAMERA_GUARDS += (
    ('one_view_narration', r'\bin\s+(?:one|a\s+single)\s+(?:view|image)\b'),
    ('qualified_image_portion', r'\b(?:upper|lower|top|bottom)\s+(?:(?:center|central|left|right)\s+)?(?:portion|part|area)\s+of\s+(?:(?:the|some|early|later|initial|final)\s+)?(?:frames?|images?|screens?|views?)\b'),
    ('intermittent_appearance_narration', r'\bappears?\s+intermittently\s+at\s+(?:the\s+)?edges?(?=[.!?;,]|$)' ),
)
OTHER_GUARDS += (
    ('suspended_checkpoint_interface',r'\b(?:transparent|translucent)\s+(?:[\w-]+\s+){0,3}checkpoint\s+panels?\b[^.!?;]{0,90}\b(?:suspended|floating|hovering)\s+in\s+(?:the\s+)?air\b'),
)

def _reviewed_v7_policy(value: Any) -> list[str]:
    if not isinstance(value,dict):return []
    errors=[];description=value.get('scene_description');dynamics=value.get('scene_dynamics')
    if isinstance(description,str):
        # Finite active water verbs do not ban static channel geography, nor
        # assert that any still whitewater/ripples actually change.
        for match in re.finditer(r'\bwater\s+(?:rushes|cascades|churns|splashes)\b',description,re.I):
            errors.append(f'scene_description: unsupported_output:finite_water_behavior_in_static_field; matched {match.group(0)!r}. Preserve supported static water appearance; independently evidenced water activity belongs only in scene_dynamics. Never infer flow from still texture or invent motion to repair this field.')
        for match in re.finditer(r'\b(?:flashing|blinking|flickering)\s+(?:[\w-]+\s+){0,3}lights?\b',description,re.I):
            errors.append(f'scene_description: unsupported_output:changing_light_behavior_in_static_field; matched {match.group(0)!r}. Preserve light appearance here; retain changing light behavior only when the complete sequence proves it independently, and only in scene_dynamics.')
        for sentence in re.split(r'[.!?;]',description):
            for match in re.finditer(r'\b(?:cars?|vehicles?|trucks?|taxis?|motorcycles?|buses)\b[^.!?;]{0,60}\b(?:moves?|drives?|travels?|collides?|crosses)\b',sentence,re.I):
                before=match.group(0).rsplit(' ',1)[0]
                if re.search(r'\b(?:where|which|that|while|whereas|people|pedestrians|person|man|woman|cyclist)\b',before,re.I):continue
                errors.append(f'scene_description: unsupported_output:finite_vehicle_behavior_in_static_field; matched {match.group(0)!r}. Keep vehicle appearance here; preserve clearly supported independent background traffic only in scene_dynamics. Omit principal/observer-controlled vehicle motion rather than moving it between fields.')
            if value.get('perspective_type')=='first_person':
                match=re.search(r'\bheld\s+by\s+(?:an?|the)\s+forearm\b',sentence,re.I)
                if match and not re.search(r'\b(?:background|statue|sculpture|painting|mannequin|person|man|woman|warrior|archer)\b',sentence,re.I):
                    errors.append(f'scene_description: camera_control:unbound_observer_forearm; matched {match.group(0)!r}. Omit unowned observer body; retain supported physical object and separately embodied background actor appearance.')
                match=re.search(r'\b(?:rifle|pistol|weapon|paddle|bow)\b[^.!?;]{0,100}\b(?:lower|bottom)\s+(?:center\s+)?foreground\b',sentence,re.I)
                if match and not re.search(r'\b(?:background|statue|sculpture|painting|mannequin|person|man|woman|warrior|archer)\b',sentence,re.I):
                    errors.append(f'scene_description: camera_control:controlled_foreground_composition; matched {match.group(0)!r}. Retain physical object appearance without observer-relative image placement or framing. Separately embodied background objects remain legal.')
    if isinstance(dynamics,str):
        for sentence in re.split(r'[.!?;]',dynamics):
            for match in re.finditer(r'\b(?:remain|remains|stay|stays)\s+(?:static|lit)\b',sentence,re.I):
                if re.search(r'\b(?:until|before|then)\b[^.!?;]{0,70}\b(?:changes?|turns?|switches?|flickers?|dims?)\b',sentence[match.end():],re.I):continue
                errors.append(f'scene_dynamics: unsupported_output:static_state_v7; matched {match.group(0)!r}. Keep supported static state in scene_description; preserve independently evidenced physical transitions and other background events.')
            if value.get('perspective_type')=='first_person':
                match=re.search(r'\b(?:the\s+)?paddle\s+blade\s+moves?\b[^.!?;]{0,80}\b(?:periphery|foreground)\b',sentence,re.I)
                if match and not re.search(r'\b(?:background|person|man|woman|paddler|kayaker)\b',sentence,re.I):
                    errors.append(f'scene_dynamics: camera_control:controlled_paddle_motion; matched {match.group(0)!r}. Omit observer-controlled paddle action and framing. Preserve explicitly distinct, independently evidenced background paddling and physical water activity.')
    return errors


CAMERA_GUARDS += (
    ('explicit_scene_image_edge', r'\b(?:left|right|upper|lower|top|bottom)\s+(?:edge|corner)\s+of\s+(?:the\s+)?scene\b'),
    ('explicit_view_composition', r'\bframes?\s+(?:the\s+)?(?:sides?|edges?)\s+of\s+(?:the\s+)?view\b|\b(?:curves?|runs?|extends?|stretches?)\b[^.!?;]{0,35}\bthrough\s+(?:the\s+)?frame\b'),
    ('away_from_foreground', r'\b(?:moves?|moving|walks?|walking|runs?|running|travels?|traveling)\b[^.!?;]{0,45}\baway\s+from\s+(?:the\s+)?foreground\b'),
    ('single_observation_narration', r'\bin\s+(?:one|a\s+single)\s+observation\b|\b(?:present|visible|absent)\s+in\s+others\b'),
    ('later_path_reveal', r'\b(?:appear|appears|visible)\s+in\s+(?:the\s+)?later\s+(?:stretches|sections|parts)\s+of\s+(?:the\s+)?path\b'),
    ('explicit_panel_composition', r'\bsplit[- ]screen\s+(?:view|layout|composition)\b|\b(?:upper|lower)\s+view\s+(?:shows|displays)\b'),
    ('observer_shadow_capture', r'\bshadow\s+of\s+(?:the\s+)?(?:camera\s+operator|observer|viewer)\b'),
)
OTHER_GUARDS += (
    ('editorial_location_overlay', r'\b(?:location\s+marker|elevation\s+information|text)\b[^.!?;]{0,90}\boverlays?\s+(?:the\s+)?scene\b'),
    ('unresolved_weather_time_alternative', r'\bovercast\s+or\s+(?:at\s+)?dusk\b|\bdusk\s+or\s+overcast\b'),
)

def _reviewed_v8_policy(value: Any) -> list[str]:
    if not isinstance(value, dict):
        return []
    errors = []
    description = value.get('scene_description')
    dynamics = value.get('scene_dynamics')
    action = r'(?:moves?|walks?|runs?|flies|fly|orbits?|flaps?|leaps?|jumps?|gestures?|waves?|crosses|cross|rises?|falls?)'
    actor_patterns = (
        (r'(?:pedestrians|people|spectators|passers|passersby|persons|men|women|characters|figures|riders|birds|seagulls)', r'(?:move|walk|run|fly|orbit|flap|leap|jump|gesture|wave|cross|rise|fall)'),
        (r'(?:person|man|woman|character|figure|rider|bird|seagull)', r'(?:moves|walks|runs|flies|orbits|flaps|leaps|jumps|gestures|waves|crosses|rises|falls)'),
    )
    if isinstance(description, str):
        for sentence in re.split(r'[.!?;]', description):
            matches = [(match, actors, verb_pattern) for actors, verb_pattern in actor_patterns for match in re.finditer(r'\b' + actors + r'\b[^.!?;]{0,70}?\b' + verb_pattern + r'\b', sentence, re.I)]
            for match, actors, verb_pattern in matches:
                verb = re.search(r'\b' + verb_pattern + r'\b$', match.group(0), re.I)
                between = match.group(0)[len(re.match(actors, match.group(0), re.I).group(0)):verb.start()]
                if re.match(r"['’]s\b", between) or re.search(r'\b(?:shadow|reflection|portrait|image|outline|silhouette)\s+of\s+(?:(?:a|an|the)\s+)?$', sentence[:match.start()], re.I):
                    continue
                # A different grammatical subject or relative clause must not
                # attribute its action to this actor. Path geography is legal.
                if re.search(r'\b(?:where|which|that|while|whereas|whose|water|river|stream|road|path|lights?|clouds?|rain|snow)\b|\band\s+(?:a|an|the)\b', between, re.I):
                    continue
                if verb.group(0).lower() in ('run', 'runs') and re.match(r'\s+of\b', sentence[match.end():], re.I):
                    continue
                errors.append(f'scene_description: unsupported_output:finite_actor_event_v8; matched {match.group(0)!r}. Keep supported actor appearance and static posture here. Put only independently evidenced BACKGROUND activity in scene_dynamics; omit principal/observer action rather than relocating it. Do not infer movement from a still depiction.')
            for match in re.finditer(r'\b(?:steam|smoke)\b(?:\s+or\s+(?:steam|smoke))?\s+(?:rises?|drifts?|billows?)\b', sentence, re.I):
                errors.append(f'scene_description: unsupported_output:finite_plume_event_v8; matched {match.group(0)!r}. Keep supported plume appearance here; put rising/drifting only in dynamics if ALL frames independently establish it. Never invent kitchen steam from reflections or common knowledge.')
            if value.get('perspective_type') == 'third_person':
                match = re.match(r'\s*(?:A|An|The)\s+(?:[\w-]+\s+){0,8}?(?:monorail\s+)?train\b[^.!?;]{0,120}?\b(?:travels?|moves?|runs?)\b', sentence, re.I)
                if match and not re.search(r'\b(?:where|which|that|while|whereas|background|another|other)\b', match.group(0), re.I):
                    errors.append(f'scene_description: principal_subject_motion:declared_train_action_v8; matched {match.group(0)!r}. Keep the declared principal train appearance without travel/action; preserve separately evidenced background water and other activity.')
        # This requires reinspection only when the answer explicitly asserts
        # active precipitation. Snow cover and wet pavement are insufficient.
        active_rain = re.search(r'\b(?:active\s+rainfall|rain\s+streaks|rain\s+(?:is\s+)?present[^.!?;]{0,30}\bvisible\s+as\s+streaks)\b', description, re.I)
        if active_rain and isinstance(dynamics, str) and not re.search(r'\b(?:rain|rainfall|precipitation)\b', dynamics, re.I):
            errors.append('scene_dynamics: observed_background_event_check:asserted_precipitation_v8. Reinspect ALL supplied frames for independent precipitation. Preserve visible falling rain in scene_dynamics if established; otherwise omit the unsupported active-rain claim and retain wetness/cloud appearance. Do not infer rain from wet ground or add it by common knowledge.')
    if isinstance(dynamics, str):
        transitions = r'\b(?:until|before|then)\b[^.!?;]{0,70}\b(?:changes?|turns?|switches?|flickers?|dims?|moves?|walks?|stands?\s+up)\b'
        for sentence in re.split(r'[.!?;]', dynamics):
            for match in re.finditer(r'\b(?:remain|remains|stay|stays)\s+(?:(?:on|in)\s+)?(?:green|red|blue|yellow|positioned|seated|standing|together|illuminated)\b', sentence, re.I):
                if re.search(transitions, sentence[match.end():], re.I):
                    continue
                errors.append(f'scene_dynamics: unsupported_output:static_posture_or_signal_v8; matched {match.group(0)!r}. Keep stable posture and signal appearance in scene_description; preserve actual independently evidenced state transitions and other background activities.')
            match = re.search(r'\b(?:a|the)\s+person\s+(?:in\s+(?:[\w-]+\s+){0,6})?(?:is\s+among|holds?)\b', sentence, re.I)
            if match and not re.search(r'\b(?:walks?|walking|moves?|moving|waves?|waving|raises?|raising|lowers?|lowering|gestures?|gesturing|shifts?|shifting|turns?|turning)\b', sentence[match.end():], re.I):
                errors.append(f'scene_dynamics: unsupported_output:static_actor_relation_v8; matched {match.group(0)!r}. Put supported person position and held-object appearance in scene_description, retaining only independently observed activity in scene_dynamics.')
    return errors

def normalize_json_envelope(value: Any) -> tuple[Any, dict[str, Any]]:
    audit = {"policy": JSON_ENVELOPE_POLICY}
    if not isinstance(value, str):
        return value, {**audit, "status": "non_text_response"}
    trimmed = value.strip(" \t\r\n")
    normalized, envelope = trimmed, "none"
    match = re.fullmatch(r"```(json)?\r?\n([\s\S]*?)\r?\n```", trimmed)
    if match and trimmed.count("```") == 2:
        normalized = match.group(2)
        envelope = "json_fence" if match.group(1) else "untyped_fence"
    return normalized, {**audit, "status": "envelope_checked", "envelope": envelope,
                        "outer_whitespace_removed": trimmed != value,
                        "input_sha256": hashlib.sha256(value.encode()).hexdigest(),
                        "json_text_sha256": hashlib.sha256(normalized.encode()).hexdigest()}


def response_schema() -> dict[str, Any]:
    properties = {name: {"type": "string"} for name in FACT_KEYS}
    properties["visual_style"]["enum"] = list(VISUAL_STYLES)
    properties["perspective_type"]["enum"] = list(PERSPECTIVE_TYPES)
    return {"type": "object", "additionalProperties": False,
            "required": sorted(FACT_KEYS), "properties": properties}


def validate_facts(value: Any) -> list[str]:
    if not isinstance(value, dict):
        return ["facts must be a JSON object"]
    errors = []
    if set(value) != FACT_KEYS:
        errors.append("facts keys must exactly match the four-field full-video H3 v3 schema")
    if value.get("perspective_type") not in PERSPECTIVE_TYPES:
        errors.append("perspective_type must equal one of: " + "; ".join(PERSPECTIVE_TYPES))
    if value.get("visual_style") not in VISUAL_STYLES:
        errors.append("visual_style must equal an allowed visual-medium enum")
    for field, (minimum, maximum) in TEXT_LIMITS.items():
        text = value.get(field)
        if not isinstance(text, str):
            errors.append(f"{field} must be a string")
            continue
        if not minimum <= len(text) <= maximum:
            errors.append(f"{field} length must be {minimum}..{maximum} characters")
        if text != text.strip() or any(ord(c) < 32 or ord(c) == 127 for c in text):
            errors.append(f"{field} must be one paragraph without control or surrounding whitespace")
        if text and not re.search(r"[A-Za-z]", text):
            errors.append(f"{field} must contain English descriptive text")
        if field == "scene_dynamics" and text.lower() in {"none", "n/a", "no motion", "no movement", "static"}:
            errors.append("scene_dynamics must be empty when no allowed dynamics are reliably observed")
        for category, guards in (("camera_control", CAMERA_GUARDS),
                                 ("principal_subject_motion", PRINCIPAL_GUARDS),
                                 ("unsupported_output", OTHER_GUARDS)):
            for name, expression in guards:
                match = re.search(expression, text, re.IGNORECASE)
                if match:
                    guidance = {
                        "camera_control": "Describe the physical scene directly; omit camera/observer control, viewing order and image-relative trajectories.",
                        "principal_subject_motion": "Keep the principal subject's supported appearance, but omit its actions and displacement; background activity remains allowed.",
                        "unsupported_output": "Use directly supported visual facts only; omit speculation, inferred audio, template labels and perspective reasoning from descriptive text.",
                    }[category]
                    errors.append(f"{field}: {category}:{name}; matched {match.group(0)!r}. {guidance}")
    return errors + _explicit_dynamic_policy(value) + _declared_principal_policy(value) + _reviewed_v3_policy(value) + _reviewed_v4_policy(value) + _reviewed_v5_policy(value) + _reviewed_v6_policy(value) + _reviewed_v7_policy(value) + _reviewed_v8_policy(value)


def _sentence(text: str) -> str:
    return text if text.endswith((".", "!", "?")) else text + "."


def serialize_h3_prompt(facts: dict[str, Any]) -> str:
    errors = validate_facts(facts)
    if errors:
        raise ValueError("invalid H3 facts: " + "; ".join(errors))
    prose = " ".join(_sentence(facts[key]) for key in
                     ("visual_style", "scene_description", "scene_dynamics") if facts[key])
    return (FIRST_FRAME_INSTRUCTION + "\n\nintegrated_multimodal_description: [Shot 1] " + prose
            + "\n\noverall_soundscape: N/A\n\nnon_diegetic_music: N/A")


def make_test_copy(source_record: Mapping[str, Any], facts: dict[str, Any],
                   provenance: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(source_record, Mapping) or not isinstance(source_record.get("caption"), str):
        raise ValueError("source must contain a string caption to preserve exactly")
    if {"h3_i2va", "h3_prompt", "h3_caption_provenance"}.intersection(source_record):
        raise ValueError("source already contains H3 fields")
    if "static_scene_description" in source_record and source_record["static_scene_description"] != source_record["caption"]:
        raise ValueError("existing static_scene_description differs from source caption")
    if not isinstance(provenance, Mapping) or not provenance:
        raise ValueError("nonempty runner provenance is required")
    prompt = serialize_h3_prompt(facts)
    result = copy.deepcopy(dict(source_record))
    result["static_scene_description"] = result.pop("caption")
    result["h3_i2va"] = {**copy.deepcopy(facts), "audio_policy": AUDIO_POLICY,
                         "motion_text_policy": MOTION_TEXT_POLICY}
    result["h3_prompt"] = prompt
    result["h3_caption_provenance"] = {**copy.deepcopy(dict(provenance)), "schema_version": SCHEMA_VERSION}
    return result
