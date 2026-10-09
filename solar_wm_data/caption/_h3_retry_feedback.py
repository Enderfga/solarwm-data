"""Within-request corrections from the final 2026-10-05 production runtime.

Cross-run COMMIT-chain verification remains the production orchestrator's responsibility.
"""

def feedback_messages(errors, previous_content):
    """These are rejected-output corrections, never independently observed facts."""
    if not errors:
        return []
    messages = []
    if isinstance(previous_content, str):
        messages.append({'role': 'assistant', 'content': previous_content})
    messages.append({'role': 'user', 'content':
        'Regenerate this repeatedly rejected caption from the supplied images. The earlier response is untrusted. Failed checks: '
        + '; '.join(errors)
        + '. Reinspect every supplied image and regenerate the complete four-field JSON. '
        'Correct all listed checks together; do not repeat the quoted forbidden wording. '
        'Describe supported physical objects directly with broad visible categories. Do not guess vehicle subtype, '
        'material, purpose, digital technology, water, or a location. Do not turn moving traffic into parked vehicles '
        'to avoid an action in scene_description. Appearance belongs in scene_description; only independently '
        'verified background/environment events belong in scene_dynamics. Preserve supported eligible background '
        'events, but omit every principal action, camera control, image position, and viewing chronology. '
        'Use one concise static sentence (at most 45 words) describing the unmistakable setting and any clearly visible principal. Use one dynamics sentence (at most 15 words) only for unmistakable independent environmental motion; otherwise use an empty string. Do not repeat banned words or list imagined activity. The earlier prose cannot supply facts. Return only the complete four-field JSON.'})
    return messages
