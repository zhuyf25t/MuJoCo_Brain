"""Learn pure motion after localization; reuse target measurements across modules."""

from .contracts import CapabilityInput, Frame, Target, direction_request


def learn_motion(state, capabilities, memory, log):
    transition = state.last_transition
    if not transition or state.learned_frame_id == state.frame.id:
        return
    state.learned_frame_id = state.frame.id
    commands = transition["commands"]
    directions = {c["tool"] for c in commands}
    if not commands or len(directions) != 1 or next(iter(directions)) not in (
            "forward", "back", "turn_left", "turn_right"):
        return
    before = Frame.model_validate(transition["before"])
    previous = Target.model_validate(transition["before_target"]) if transition.get("before_target") else None
    result = capabilities.call("analyze_motion", CapabilityInput(
        frame=state.frame, comparison=before, commands=commands,
        target=state.target, comparison_target=previous))
    observation = None
    if result.source == "shared_target":
        # The model only decides whether the SAME unmoved object is a usable reference.
        # Coordinates are copied from the two recorded detections, never re-estimated.
        if result.valid and (previous is None or state.target is None):
            raise ValueError("shared_target analysis requires both frame detections")
        if result.valid:
            observation = {"before": previous.model_dump(), "after": state.target.model_dump(),
                           "before_frame_id": before.id, "after_frame_id": state.frame.id}
            result = result.model_copy(update={"before": previous.center, "after": state.target.center})
    sample = memory.add(direction_request(commands[0]["tool"]), [before, state.frame],
                        commands, result, observation=observation)
    log.append("motion_sample", accepted=sample is not None,
               sample_id=sample["id"] if sample else None, source=result.source,
               analysis=result.model_dump(), observation=observation)
